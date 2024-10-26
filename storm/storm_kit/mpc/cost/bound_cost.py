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
import torch
import torch.nn as nn
# import torch.nn.functional as F
from .gaussian_projection import GaussianProjection
from BGU.Rlpt.DebugTools.storm_tools import RealWorldState, is_real_world
from BGU.Rlpt.Classes.CostTerm import CostTerm
from BGU.Rlpt.DebugTools.globs import GLobalVars


class BoundCost(nn.Module):
    def __init__(self, tensor_args={'device':torch.device('cpu'), 'dtype':torch.float64},
                 bounds=[], weight=1.0, gaussian_params={}, bound_thresh=0.1):
        super(BoundCost, self).__init__()
        self.tensor_args = tensor_args
        self.weight = torch.as_tensor(weight, **tensor_args)
        self.proj_gaussian = GaussianProjection(gaussian_params=gaussian_params)

        self.bounds = torch.as_tensor(bounds, **tensor_args)
        self.bnd_range = (self.bounds[:,1] - self.bounds[:,0]) / 2.0
        self.t_mat = None
        self.bound_thresh = bound_thresh * self.bnd_range
        self.bounds[:,1] -= self.bound_thresh
        self.bounds[:,0] += self.bound_thresh
    def forward(self, state_batch):
        inp_device = state_batch.device

        bound_mask = torch.logical_and(state_batch < self.bounds[:,1],
                                       state_batch > self.bounds[:,0])

        cost = torch.minimum(torch.square(state_batch - self.bounds[:,0]),torch.square(self.bounds[:,1] - state_batch))
        
        cost[bound_mask] = 0.0

        cost = (torch.sum(cost, dim=-1))
        # cost = self.weight * self.proj_gaussian(torch.sqrt(cost))
        
        w1 = self.weight # Dan
        t1 = self.proj_gaussian(torch.sqrt(cost)) # Dan
        cost = w1 * t1 # Dan

        sniffer = GLobalVars.cost_sniffer
        if sniffer.is_initialized():      
            sniffer.set('state_bound', CostTerm(w1, t1))
        # cost = self.weight * cost 
        

        
        
        return cost.to(inp_device)
    def update_weight(self, weight):
        """
        Update weight dynamically
        """
        self.weight = torch.as_tensor(weight, **self.tensor_args)
