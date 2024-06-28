import copy
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
import random
import time
import threading
import webbrowser


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
        
        # Store the latest data to be displayed
        self.latest_data = {}

        @self.app.callback(Output('live-graph', 'figure'),
                           [Input('interval-component', 'n_intervals')])
        def update_graph_live(n):
            if not self.latest_data:
                return go.Figure()

            x = list(self.latest_data.keys())
            y = list(self.latest_data.values())
            fig = go.Figure(data=[go.Scatter(x=x, y=y, mode='lines+markers')])
            return fig

    def _update_live_gui_with(self, data):
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
            self._gui_dashboard1 = Gui(title="Real World Costs", port=8050)
            self._gui_dashboard2 = Gui(title="MPC Costs", port=8051)
            self._real_world_heatmap_thread = threading.Thread(target=self._update_heatmap, daemon=True, kwargs={"real_world": True})
            self._mpc_heatmap_thread = threading.Thread(target=self._update_heatmap, daemon=True, kwargs={"real_world": False})
            self._real_world_heatmap_thread.start()
            self._mpc_heatmap_thread.start()
            threading.Thread(target=self._gui_dashboard1.run, daemon=True).start()
            threading.Thread(target=self._gui_dashboard2.run, daemon=True).start()
            # URLs to open
            self._real_world_url = "http://127.0.0.1:8050"
            self._mpc_url = "http://127.0.0.1:8051"
            webbrowser.open(self._real_world_url)
            webbrowser.open(self._real_world_url)


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

        print("WRITER DEBUG")
        print(f"WRITING TO {'real world array' if lock == lock1 else 'mpc array'} with len {len(dst_array)}")

    def _update_heatmap(self, real_world: bool):  # thread target
        read_from_real_world = real_world

        while True:
            dst_array =  self.costs_real if real_world else self.costs_mpc
            if len(dst_array):
                
                if read_from_real_world:
                    lock = lock1
                    gui = self._gui_dashboard1
                    
                else:
                    lock = lock2
                    gui = self._gui_dashboard2
                
                raw_display_data = dst_array[-1]    
                display_data = {ct_name: ct.mean() for ct_name, ct in raw_display_data.items()} # convet each CostTerm to its mean over nxk rollouts x horizons (int he mpc case) or leaves it the same (mean of single value) in the real world case   
                with lock:
                    if len(dst_array):
                        print(f"READING FROM {'real world array' if lock == lock1 else 'mpc array'} with len {len(dst_array)}") 
                        gui._update_live_gui_with(display_data)
                        
                
            time.sleep(0.1)

# Example usage:
# cost_fn_sniffer = CostFnSniffer(gui=True)
# # Add some test data
# cost_fn_sniffer.set("test_cost", CostTerm())  # Add real cost data here
# cost_fn_sniffer.finish()
