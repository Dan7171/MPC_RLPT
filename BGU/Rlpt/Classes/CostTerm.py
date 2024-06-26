
from dataclasses import dataclass
from typing import List, Union,Iterable,Collection
import numpy as np
import torch

class CostTerm:
    
    def __init__(self,weight:torch.Tensor, term:torch.Tensor):
        self.weight = weight # this is its weight 
        self.term = term # this is the unweighted cost 
        self.cost: torch.Tensor = weight * term # the final cost 
    
    def __add__(self, other: 'CostTerm'): # overide +
        return self.cost + other.cost
    
    @classmethod
    def sum(cls,cost_terms: Iterable['CostTerm']) -> torch.Tensor:
        tensor_sum: torch.Tensor = None
        for ct in cost_terms:
            if tensor_sum is None:
                tensor_sum = ct.cost
            else:
                tensor_sum += ct.cost
        return tensor_sum
    
        
    # def total_weighted(self):
    #     return self.weights @ self.terms
     
    
     
    
     