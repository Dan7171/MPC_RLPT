from dataclasses import dataclass
from datetime import datetime
import os
from typing import List
import yaml
from BGU.Rlpt.DebugTools.CostFnSniffer import CostFnSniffer

##### Globals ######
class GLobalVars:
    cost_sniffer:CostFnSniffer # if using sniffer, will be changed from None to the sniffer
    print(f"\n\n\n\n\n\n\n SNIFFER \n\n\n\n\n\n\n\n\n")
