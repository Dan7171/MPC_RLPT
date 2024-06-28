
from dataclasses import dataclass
from typing import List, Union,Iterable,Collection
import numpy as np
import torch

class CostTerm:
    """
    
    Represent data of a unqiue "cost term" from the simualpor/paper(for example, "manipulability cost"), from a specific "step".
    Each "step" corresponds to a unique (real-world) time-unit at the simulator - "t".
    The "step", and by that the CostTerm, can be one of the two:
        1. a "planning" step of time "t" - taken by the  "mpc" => the data is a collection of actions and rollouts.
        2. a "real world" step, will contain only one action, which has been chosen after the planning step of the mpc, and sent to the (real) controller. 
    
    Recap:
        1. CostTerm instance is data from a "step".
        2. a "step" is either a planning (by mpc) or real (by controller) step.
         
    What a CostTerm is composed from?
    
    Each "step" generates (and thats what we are saving in CostTerm) a "sequence" of n >= 1 serieses of actions, with k>=1 actions per series.     
    
    Reminder: the weighted cost is evaluated for each action in each series!
    
    In the planning-case (mpc):
        - the "sequence" is the series of n(>=1, normally >1) rollouts (particles)
        - each "series" in the sequence has a fixed length of k(>=1, normally >1) (horizon) actions .
    
    In the "real" (controller case, not mpc):
        - n=1 (the "sequence" is only has one "series" of actions)
        - k=1 the single "series" has only one action. 
        
    
    """
    def __init__(self,weight:Union[float,torch.Tensor], term:torch.Tensor):
        """ Initializing the CostTerm. 
             
        Args:
            weight (Union[float,torch.Tensor]): The wait of the cost term, where each term of an action  is multiplied by. Normally just  a float 
            
            term (torch.Tensor): the unweighterd cost terms. Formally: 
                
                If the CostTerm represents an mpc "step":
                    term[i][j] = unweighted cost of the jth action at the ith rollout.
                
                If the CostTerm represents a real world "step":
                    term[i][j] = costTerm[0][0] - just the cost of the action which has been sent to the controller. Based on the result in the real world
            
                      
        """
        self.weight = weight  
        self.term = term 
        self.cost: torch.Tensor = weight * term # the final cost 
    
    # def __add__(self, other: 'CostTerm'): # overide +
    #     return self.cost + other.cost
    
    @classmethod
    def sum(cls,cost_terms: Iterable['CostTerm']) -> torch.Tensor:
        tensor_sum: torch.Tensor = None
        for ct in cost_terms:
            if tensor_sum is None:
                tensor_sum = ct.cost
            else:
                tensor_sum += ct.cost
        return tensor_sum
    
    def _get_tensor_mean(self, t:torch.Tensor, type=None):
        
        m:torch.Tensor = torch.mean(t)
        
        if type == np.float64:
            return np.float64(m)
        
        return m
    
    def mean(self) -> np.float64:
        """Get the mean cost of the Cost-Term (tensor of ).

        Returns:
            np.float64: _description_
        """
        return self._get_tensor_mean(self.cost, type=np.float64)

    
    def mean_term(self) -> np.float64:
        """Get the mean of the terms 

        Returns:
            np.float64: _description_
        """
        return self._get_tensor_mean(self.term, dtype=np.float64)
    
    # def total_weighted(self):
    #     return self.weights @ self.terms
     
    
     
    
     