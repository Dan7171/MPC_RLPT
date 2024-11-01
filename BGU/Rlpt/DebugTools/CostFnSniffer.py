import copy
import os
import pickle
import subprocess
import time
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import torch
from BGU.Rlpt.DebugTools.storm_tools import RealWorldState, is_real_world 
from BGU.Rlpt.Classes.CostTerm import CostTerm
import threading
import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import time
import webbrowser
import gc

local_address = "http://127.0.0.1"

# Locks for each list
lock1 = threading.Lock()
lock2 = threading.Lock()

class Gui:
    def __init__(self, title, port):
        
        self.latest_data = {}
        self.axis_names = []
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
        
        @self.app.callback(Output('live-graph', 'figure'),
                           [Input('interval-component', 'n_intervals')])
        
        def update_graph_live(n): # auitomatically called by the callback
            if not self.latest_data:
                return go.Figure()
                # return go.Bar()
            term_names = list(self.latest_data.keys())
            ys = self.latest_data.values()
            costs = [v[0] for v in ys], self.axis_names[1]
            weights = [v[1] for v in ys], self.axis_names[2]
            weights_only = [float(t) for t in weights[0]] 
            
            show_weights = True
            data = [go.Scatter(x=term_names, y=costs[0], mode='lines+markers', name=costs[1])]
            if show_weights:
                data.append(go.Scatter(x=term_names, y=weights_only, mode='lines+markers',name=weights[1]))
            fig = go.Figure(data=data)
            # fig = go.Figure(data=[go.Bar(x=x, y=y)])
            
            return fig

    def update(self, data, axis_names):
        # lock = lock1 if is_real_world() else lock2
        # with lock:
        #     self.latest_data = data
        #     if not self.axis_names is None:
        #         self.axis_names = axis_names
        self.latest_data = data
        if not self.axis_names is None:
            self.axis_names = axis_names
            

    def run(self):
        self.app.run_server(debug=True, use_reloader=False, port=self.port)

class CostFnSniffer:
    def __init__(self, gui=False, save_costs=False, buffer_n=1000):
        
        self._current_ts_costs_real = {} # Cuurent timestep costs (an array which is switching between real world and  mpc at a time). We fill it with costs along the cost fn and wipe it afterwards.- will be wiped at the beginning of every cost_fn calc
        self._current_ts_costs_mpc = {} # Cuurent timestep costs (an array which is switching between real world and  mpc at a time). We fill it with costs along the cost fn and wipe it afterwards.- will be wiped at the beginning of every cost_fn calc
        self.save_costs = save_costs # save all costs to a file
        self.gui = gui
        self.is_contact_real_world = False # if collision/contact with obstacles was detected
        self.mppi_policy = [torch.Tensor(), torch.Tensor()] #  
        # relevant only when save_costs is True:
        if self.save_costs:    
            self.costs_buff_real = []  # buffer to store real costs which were calculated during the simulation before flushing them to storage  
            self.costs_buff_mpc = []   # buffer to store mpc costs which were calculated during the simulation before flushing them to storage
            self.buffer_n = buffer_n # the max num of items on each buffer. The larger n is, the less "flush to storage" calls (but longer writing to storage on each time).
            self.all_costs_file = os.path.join(os.getcwd(), 'costs.pickle') 
            
     
        if self.gui:           
            
            # subprocess.Popen(f'firefox {local_address}:0000', shell=True)

            # 2 GUI objects - realated to URLS
            self._gui_dashboard1 = Gui(title="Real World Costs", port=8050)
            self._gui_dashboard2 = Gui(title="MPC Costs", port=8051)
            
            
            # 2 gui dashboard threads - update gui in background
            threading.Thread(target=self._gui_dashboard1.run, daemon=True).start()
            threading.Thread(target=self._gui_dashboard2.run, daemon=True).start()
            
            # Read from costs lists
            threading.Thread(target=self._cost_terms_reading_loop, daemon=True, kwargs={"real_world": True}).start()
            threading.Thread(target=self._cost_terms_reading_loop, daemon=True, kwargs={"real_world": False, "full_horizon": True}).start()
            
    def aquire_lock(self):
        lock = lock1 if is_real_world() else lock2
        return lock
    
    def set(self, ct_name: str, ct: CostTerm):
        with self.aquire_lock():    
            is_real = is_real_world()
            costs = self._current_ts_costs_real if is_real else self._current_ts_costs_mpc
            costs[ct_name] = ct
        
    def get_current_costs(self):
        return self._current_ts_costs_real if is_real_world() else self._current_ts_costs_mpc
                
    def _flush_to_storage(self, buff, real_world):
        pickle_file = self.all_costs_file
        
        # Check if the file exists and is non-empty
        if os.path.exists(pickle_file) and os.path.getsize(pickle_file) > 0:
            # Load the existing pickle file
            with open(pickle_file, 'rb') as f:
                costs = pickle.load(f)
        else:
            # Initialize a new dict if the file is empty or doesn't exist
            costs = {'real': [], 'mpc': []}

        # Append to the correct list based on the 'real' parameter
        if real_world:
            costs['real'].extend(buff)
        else:
            costs['mpc'].extend(buff)

        # Save the updated dictionary back to the pickle file
        with open(pickle_file, 'wb') as f:
            pickle.dump(costs, f)
        
        # optinal
        del buff
        gc.collect()

    def finish(self):
        
        if self.save_costs:
            real_world = is_real_world()
            buff = self.costs_buff_real if real_world else self.costs_buff_mpc
            costs = self._current_ts_costs_real if real_world else self._current_ts_costs_mpc
            with self.aquire_lock():        
                if len(buff) == self.buffer_n:
                    self._flush_to_storage(buff, real_world)
                    buff = []
                    if real_world:    
                        self.costs_buff_real = buff
                    else:
                        self.costs_buff_mpc = buff
                    buff.append(copy.deepcopy(costs))
        
    
        
    def _cost_terms_reading_loop(self, real_world: bool,full_horizon=False):  # thread target
        update_rate = 0.1 # how often rendering you
        time.sleep(update_rate)
        while True:
            gui = self._gui_dashboard1 if is_real_world else self._gui_dashboard2 
            # all_costs =  self.costs_buff_real if real_world else self.costs_buff_mpc
            with self.aquire_lock():
                costs_to_show = self._current_ts_costs_real if real_world else self._current_ts_costs_mpc
                costs_to_show_tmp = copy.deepcopy(costs_to_show)
         
                if len(costs_to_show): # wait for first write to dest array        
                    if full_horizon:
                        # Option 1 - Display at the mpc graph the mean cost over all trajectories
                        display_data = {ct_name: (ct.mean(), ct.weight) for ct_name, ct in costs_to_show_tmp.items()} # convet each CostTerm to its mean over nxk rollouts x horizons (int he mpc case) or leaves it the same (mean of single value) in the real world case   
                    else:
                        # Option 2 - display the mean cost only over the first actions in rollouts
                        # Display at the mpc graph the mean cost over all trajectories
                        display_data = {ct_name: (ct.mean_over_first_action(),ct.weight) for ct_name, ct in costs_to_show_tmp.items()}
                        
                
                    x = 'cost term'
                    y1 = f'mean weighted cost (all rollouts, '+ ('whole horizon)' if full_horizon else 'first action in horizon)')
                    y2 = f'weight'
                    gui.update(display_data,(x,y1,y2))
                        
            
            time.sleep(update_rate)
            
    def get_current_mppi_policy(self):
        return self.mppi_policy
    
    def set_current_mppi_policy(self, horizon_means, covariances):
        if horizon_means is not None:
            self.mppi_policy[0] = horizon_means
        if covariances is not None:
            self.mppi_policy[1] = covariances        


# Example usage:
# cost_fn_sniffer = CostFnSniffer(gui=True)
# # Add some test data
# cost_fn_sniffer.set("test_cost", CostTerm())  # Add real cost data here
# cost_fn_sniffer.finish()
