import copy
import time

from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
from BGU.Rlpt.DebugTools.storm_tools import RealWorldState, is_real_world 
from BGU.Rlpt.Classes.CostTerm import CostTerm
import threading
# lock = threading.Lock()

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
            self._ax = heatmaps[0]
            self._sm = heatmaps[1]
            display_thread = threading.Thread(target=self._update_heatmap)
            display_thread.daemon = True
            display_thread.start()
            
    
        
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
        target = self.costs_real if is_real_world() else self.costs_mpc
        target.append(copy.deepcopy(self._buffer))
        

    def _setup_heatmap(self):
        
        plt.ion()
        # Set up the color map
        fig, ax = plt.subplots()
        bars = ax.bar(['a', 'b', 'c'], [0, 0, 0], color='green')
        norm = Normalize(vmin=0, vmax=100)
        cmap = plt.get_cmap('RdYlGn_r')
        sm = ScalarMappable(norm=norm, cmap=cmap)
        return ax, sm
      
    # Function to read and display the shared data
    def _update_heatmap(self):
        
        # plt.ion()
        # # bars = ['a','b','c']
        # bars = self._ax.bar(['a', 'b', 'c'], [0, 0, 0], color='green')
        # while True:
        #     values = np.random.rand(3)
        #     for bar, value in zip(bars, values):
        #         bar.set_height(value)
        #         bar.set_color(self._sm.to_rgba(value))
        #     # self._ax.set_ylim(0, 100)  # Ensure the y-axis always goes from 0 to 100
        #     plt.draw()
        #     plt.pause(0.01)
        #     time.sleep(0.01)
        
        while True:
            print("hiu")
            time.sleep(1)
        