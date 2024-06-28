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

# Locks for each list
lock1 = threading.Lock()
lock2 = threading.Lock()


class CostFnSniffer:
    def __init__(self, show_heatmap=True):
        self.costs_real = [] # all real costs which were calculated during the simulation  
        self.costs_mpc = [] # all mpc costs which were calculated during the simulation
        self._buffer = {} # We fill it with costs along the cost fn and wipe it afterwards.- will be wiped at the begninning of every cost_fn calc, 
        self.show_heatmap = show_heatmap
        self._hm_real = None
        self._hm_mpc = None
        if self.show_heatmap:
            heatmaps = self._setup_heatmap()
            self._real_world_heatmap_thread = threading.Thread(target=self._update_heatmap,daemon=True, kwargs={"real_world": True}) 
            self._mpc_heatmap_thread = threading.Thread(target=self._update_heatmap, daemon=True, kwargs={"real_world": False}) 
            self._real_world_heatmap_thread.start()
            self._mpc_heatmap_thread.start()
            
    def _empty_buffer(self):
        self._buffer.clear()
        
    def _set_in_buffer(self,k,v):
        self._buffer[k] = v
    
    def set(self, ct_name:str, ct: CostTerm):
        self._buffer[ct_name] = ct
        
    def finish(self):
        self._flush_buffer()
        self._empty_buffer()
        # if self.show_heatmap:
        #     self._update_heatmap()
        
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

    def _setup_heatmap(self):
        return
      
    # Function to read and display the shared data
    def _update_heatmap(self,real_world:bool): # thread target
        read_from_real_world = real_world
        
        while True:
            if read_from_real_world:
                dst_array = self.costs_real
                lock = lock1
            else:
                dst_array = self.costs_mpc
                lock = lock2    
            
            
            # target, lock = self.costs_real, lock1 if is_real_world() else self.costs_mpc, lock2
            with lock:
                if len(dst_array):
                    print("THREAD TARGET")
                    print(f"READING FROM {'real world array' if lock == lock1 else 'mpc array'} with len {len(dst_array)}")
           
            
            
            time.sleep(0.1)
                    