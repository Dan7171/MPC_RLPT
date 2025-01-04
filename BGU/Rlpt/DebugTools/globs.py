from dataclasses import dataclass
from datetime import datetime
import os
from typing import List
import yaml
from typing import Optional, Dict
from BGU.Rlpt.DebugTools.CostFnSniffer import CostFnSniffer

##### Globals ######
class GLobalVars:
    cost_sniffer: Optional[CostFnSniffer] = None  # if using sniffer, will be changed from None to the sniffer
    rlpt_cfg: Optional[Dict] = None
    
    @staticmethod
    def is_defined(class_attribute_name: str) -> bool:
        return hasattr(GLobalVars, class_attribute_name) and getattr(GLobalVars, class_attribute_name) is not None
