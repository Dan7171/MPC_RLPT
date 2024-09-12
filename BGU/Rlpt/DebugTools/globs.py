from dataclasses import dataclass
from datetime import datetime
import os
from typing import List

import yaml
from BGU.Rlpt.DebugTools.CostFnSniffer import CostFnSniffer

##### Globals ######
@dataclass
class Globs:
    """ Global variables. Put your global variables here
    """
    
    cfg_path = 'BGU/Rlpt/Run/configs/main.yml'
    cfg = None
    with open(cfg_path) as file:
        cfg = yaml.load(file, Loader=yaml.FullLoader)

    if 'cost_sniffer' in cfg:        
        cost_fn_sniffer: CostFnSniffer = CostFnSniffer(*cfg['cost_sniffer'])
        pass
    


globs = Globs() # One initiation per process (not re-initiation on every import). Import this 
