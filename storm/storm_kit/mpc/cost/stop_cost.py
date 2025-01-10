#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.#
from sympy import limit
import torch
import torch.nn as nn
# import torch.nn.functional as F
from .gaussian_projection import GaussianProjection
from BGU.Rlpt.DebugTools.storm_tools import RealWorldState, is_real_world
from BGU.Rlpt.Classes.CostTerm import CostTerm
from BGU.Rlpt.DebugTools.globs import GLobalVars


class StopCost(nn.Module):
    def __init__(self, tensor_args={'device':torch.device('cpu'), 'dtype':torch.float64},
                 max_limit=None, max_nlimit=None, weight=1.0, gaussian_params={},
                 traj_dt=None,**kwargs):
        super(StopCost, self).__init__()
        self.tensor_args = tensor_args
        self.weight = torch.as_tensor(weight, **tensor_args)
        self.proj_gaussian = GaussianProjection(gaussian_params=gaussian_params)
        self.traj_dt = traj_dt
        self.max_nlimit = max_nlimit # max acceleration (If this is stop_cost cost term)
        self.max_limit = max_limit # (If thisis stop_cost_acc cost term. Originally unused)
        
        # compute max velocity across horizon:
        self.horizon = self.traj_dt.shape[0]
        sum_matrix = torch.tril(torch.ones((self.horizon, self.horizon), **self.tensor_args)).T

        if(max_nlimit is not None): # Guess it means stop_cost (of velocities. see /content/configs/mpc/franka_reacher.yml
            # every timestep max acceleration:
            sum_matrix = torch.tril(torch.ones((self.horizon, self.horizon), **self.tensor_args)).T
            delta_vel = self.traj_dt * max_nlimit
            self.max_vel = ((sum_matrix @ delta_vel).unsqueeze(-1))
            
        elif(max_limit is not None): # Guess it means stop_acc cost (of accelerations). see /content/configs/mpc/franka_reacher.yml
            sum_matrix = torch.tril(torch.ones((self.horizon, self.horizon), **self.tensor_args)).T
            delta_vel = torch.ones_like(self.traj_dt) * max_limit
            self.max_vel = ((sum_matrix @ delta_vel).unsqueeze(-1))
    
    def update_all(self, stop_cost_params, is_stop_acc=False):
        new_weight, new_max_limit, =  stop_cost_params
        self.weight = new_weight
        if is_stop_acc: # stop_cost_acc
            self.max_limit = new_max_limit 
        else: # stop_cost
            self.max_nlimit = new_max_limit
    
        
        
        
    def forward(self, vels,is_stop_acc=False):
        """
        Compute cost of not obeying max velocities (if not is_stop_acc) or max accelerations (if is_stop_acc).
        For each step h in horizon, we compute maximal acceleration/velocity allowed for it (in absolute value).
        We then compare it to the real velocity/acceleration in action h in a given sampled sequence of actions (a1,...aH).
        We punish (charge cost) higher, as more we exceed the max value of time h. 
        
        
        """
        # max velocity threshold:
        # H velocities of time t. if actual abs val of velocity of a(t,h) (|vel of a(t,h)|)) is greater than abs(max vel of step h), 
        # this term would be greater and the cost term would grow.
        # shortly: the greater
        
        cost_term_name = 'stop_acc' if is_stop_acc else 'stop' # this function can be used for punishg on both crossing max velocity and both crossing max acceleration
        
        inp_device = vels.device
        vel_abs = torch.abs(vels.to(**self.tensor_args)) # abs val of H velocities of time t: (|vel(a(t,h))|))
        vel_abs = vel_abs - self.max_vel  # Here we compute the errors, by how much we exceeded the max velocity of step h. For each velocity h in horizon, reduce the max velocity allowed for time step h.
        vel_abs[vel_abs < 0.0] = 0.0 # if diff < 0 (meaning abs val of actual velocity < max allowed), set 0. Otherwise, leave the difference as is        
        w1 = self.weight  # how hard we punish on errors (crossing max allowed velocity (in absolute values).
        t1 =  self.proj_gaussian(((torch.sum(torch.square(vel_abs), dim=-1)))) # square the errors (the exceeding in max velocity) 
        cost = w1 * t1    
            
        
        sniffer = GLobalVars.cost_sniffer
        # if sniffer.is_initialized():
        if sniffer is not None:     
            sniffer.set(cost_term_name, CostTerm(w1, t1))
             
        return cost.to(inp_device)
    
    def set_limit_for_stop_cost(self, max_nlimit):
        """Update max acceleration (only in  stop_cost).

        Args:
            max_nlimit (_type_): numeric. Non negative
        """
        # every timestep max acceleration:
        sum_matrix = torch.tril(torch.ones((self.horizon, self.horizon), **self.tensor_args)).T
        delta_vel = self.traj_dt * max_nlimit
        self.max_vel = ((sum_matrix @ delta_vel).unsqueeze(-1))

   
    def update_weight(self, weight):
        """
        Update weight dynamically
        """
        self.weight = torch.as_tensor(weight, **self.tensor_args)
     
    def update_max_vel(self, traj_dt):
        """
        Update traj_dt and all ots dependencies
        """
        self.traj_dt = traj_dt
        # compute max velocity across horizon:
        self.horizon = self.traj_dt.shape[0]
        sum_matrix = torch.tril(torch.ones((self.horizon, self.horizon), **self.tensor_args)).T

        if(self.max_nlimit is not None):
            # every timestep max acceleration:
            sum_matrix = torch.tril(torch.ones((self.horizon, self.horizon), **self.tensor_args)).T
            delta_vel = self.traj_dt * self.max_nlimit
            self.max_vel = ((sum_matrix @ delta_vel).unsqueeze(-1))
            
        elif(self.max_limit is not None):
            sum_matrix = torch.tril(torch.ones((self.horizon, self.horizon), **self.tensor_args)).T
            delta_vel = torch.ones_like(self.traj_dt) * self.max_limit
            self.max_vel = ((sum_matrix @ delta_vel).unsqueeze(-1))
