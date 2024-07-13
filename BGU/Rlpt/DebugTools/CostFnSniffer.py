import copy
import os
import subprocess
import time
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
from BGU.Rlpt.DebugTools.storm_tools import RealWorldState, is_real_world 
from BGU.Rlpt.Classes.CostTerm import CostTerm
import threading
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import time
import webbrowser

local_address = "http://127.0.0.1"

# Locks for each list
lock1 = threading.Lock()
lock2 = threading.Lock()

class Gui:
    def __init__(self, title, port):
        self.app = dash.Dash(__name__)
        self.port = port
        self.app.layout = html.Div([
            html.H1(title),
            dcc.Graph(id='live-graph'),
            dcc.Interval(
                id='interval-component',
                interval=1*1000,  # in milliseconds
                n_intervals=0
            )
        ])
        # run browser
        
        subprocess.Popen(f'firefox {local_address}:{port}', shell=True)

        
        # Store the latest data to be displayed
        self.latest_data = {}
            
        @self.app.callback(Output('live-graph', 'figure'),
                           [Input('interval-component', 'n_intervals')])
        
        def update_graph_live(n): # auitomatically called by the callback
            if not self.latest_data:
                return go.Figure()
                # return go.Bar()
            x = list(self.latest_data.keys())
            y = list(self.latest_data.values())
            fig = go.Figure(data=[go.Scatter(x=x, y=y, mode='lines+markers'),go.Scatter(x=x, y=[y1 + 10 for y1 in y], mode='lines+markers')])
            # fig = go.Figure(data=[go.Bar(x=x, y=y)])
            
            return fig

    def update(self, data):
        self.latest_data = data

    def run(self):
        self.app.run_server(debug=True, use_reloader=False, port=self.port)

class CostFnSniffer:
    def __init__(self, gui=True):
        self.costs_real = []  # all real costs which were calculated during the simulation  
        self.costs_mpc = []   # all mpc costs which were calculated during the simulation
        self._buffer = {}     # We fill it with costs along the cost fn and wipe it afterwards.- will be wiped at the beginning of every cost_fn calc
        self.gui = gui
        self._hm_real = None
        self._hm_mpc = None
        
        if self.gui:           
            
            # subprocess.Popen(f'firefox {local_address}:0000', shell=True)

            # 2 GUI objects - realated to URLS
            self._gui_dashboard1 = Gui(title="Real World Costs", port=8050)
            self._gui_dashboard2 = Gui(title="MPC Costs", port=8081)
            
            
            # 2 gui dashboard threads - update gui in background
            threading.Thread(target=self._gui_dashboard1.run, daemon=True).start()
            threading.Thread(target=self._gui_dashboard2.run, daemon=True).start()
            
            # Read from costs lists
            threading.Thread(target=self._cost_terms_reading_loop, daemon=True, kwargs={"real_world": True}).start()
            threading.Thread(target=self._cost_terms_reading_loop, daemon=True, kwargs={"real_world": False}).start()
            

    def _empty_buffer(self):
        self._buffer.clear()

    def _set_in_buffer(self, k, v):
        self._buffer[k] = v

    def set(self, ct_name: str, ct: CostTerm):
        self._buffer[ct_name] = ct

    def finish(self):
        self._flush_buffer()
        self._empty_buffer()
    
    def _flush_buffer(self):
        flush_to_real_world = is_real_world()
        if flush_to_real_world:
            dst_array = self.costs_real
            lock = lock1
        else:
            dst_array = self.costs_mpc
            lock = lock2

        with lock:
            dst_array.append(copy.deepcopy(self._buffer))

        
    def _cost_terms_reading_loop(self, real_world: bool):  # thread target
        read_from_real_world = real_world
        update_rate = 0.1
        while True:
            dst_array =  self.costs_real if real_world else self.costs_mpc
            if len(dst_array): # wait for first write to dest array        
                if read_from_real_world:
                    lock = lock1
                    gui = self._gui_dashboard1
                    
                else:
                    lock = lock2
                    gui = self._gui_dashboard2
                
                raw_display_data = dst_array[-1]    
                
                # Option 1 - Display at the mpc graph the mean cost over all trajectories
                # display_data = {ct_name: ct.mean() for ct_name, ct in raw_display_data.items()} # convet each CostTerm to its mean over nxk rollouts x horizons (int he mpc case) or leaves it the same (mean of single value) in the real world case   
                
                # Option 2 - display the mean cost only over the first actions in rollouts
                # Display at the mpc graph the mean cost over all trajectories
                display_data = {ct_name: ct.mean_over_first_action() for ct_name, ct in raw_display_data.items()}
                
                
                with lock:
                    if len(dst_array):
                        gui.update(display_data)
                        
            
            time.sleep(update_rate)

# Example usage:
# cost_fn_sniffer = CostFnSniffer(gui=True)
# # Add some test data
# cost_fn_sniffer.set("test_cost", CostTerm())  # Add real cost data here
# cost_fn_sniffer.finish()
