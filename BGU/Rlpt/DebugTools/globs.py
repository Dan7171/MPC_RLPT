from dataclasses import dataclass
from typing import List
from BGU.Rlpt.DebugTools.CostFnSniffer import CostFnSniffer

##### Globals ######
@dataclass
class Globs:
    """ Global variables. Put your global variables here
    """
    cost_fn_sniffer: CostFnSniffer = CostFnSniffer()

globs = Globs() # One initiation per process (not re-initiation on every import). Import this 
