from dataclasses import dataclass
from datetime import datetime
import os
from typing import List
import yaml


##### Globals ######
class GLobalVars:
    cost_sniffer = None # if using sniffer, will be changed from None to the sniffer
    
