
import math
import os
import random
from collections import deque
from typing import Deque, Dict, List, Tuple
import gymnasium as gym
from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
from torch.nn.utils import clip_grad_norm_
from BGU.Rlpt.drl.rainbow_rlpt.noisy_linear import NoisyLinear
from BGU.Rlpt.utils.type_operations import torch_tensor_to_ndarray
# from segment_tree import MinSegmentTree, SumSegmentTree

class Network(nn.Module):
    def __init__(
        self, 
        in_dim: int, 
        out_dim: int, 
        atom_size: int, 
        support: torch.Tensor,
        debug_mode=False
    ):
        """Initialization."""
        super(Network, self).__init__()
        
        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        # set common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, 512), 
            nn.ReLU(),
        )
        self.feature_layer2 = nn.Sequential(
            nn.Linear(512, 1024), 
            nn.ReLU(),
        )
        self.feature_layer3 = nn.Sequential(
            nn.Linear(1024, 1024), 
            nn.ReLU(),
        )
        self.feature_layer4 = nn.Sequential(
            nn.Linear(1024, 1024), 
            nn.ReLU(),
        )
        self.feature_layer5 = nn.Sequential(
            nn.Linear(1024, 512), 
            nn.ReLU(),
        )
        self.feature_layer6 = nn.Sequential(
            nn.Linear(512, 128), 
            nn.ReLU(),
        )
        
        
        # set advantage layer
        self.advantage_hidden_layer = NoisyLinear(128, 128)
        self.advantage_layer = NoisyLinear(128, out_dim * atom_size)

        # set value layer
        self.value_hidden_layer = NoisyLinear(128, 128)
        self.value_layer = NoisyLinear(128, atom_size)

        self.debug_mode = debug_mode # new
        if self.debug_mode:
            self.plot_ticks = 5
            self.steps = 0
            self.fig = plt.figure()
            plt.ion()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        dist = self.dist(x) # a tensor of (1 X A X atom-size) for each action in A,  dist[a][i] is the chance to get the ith q value (possible q values are linearly spaced between v min and v max in the "support" verctor) 
        
        if self.debug_mode:
            self.steps += 1
            if self.steps % self.plot_ticks == 0:
                self.fig.clear()
                plt.xlabel('v')
                plt.ylabel('pr(q(s,a)) = v')
                for action_id in range(dist.shape[1]):        
                    q_a_hist = dist[0][action_id]
                    plt.bar(torch_tensor_to_ndarray(self.support),q_a_hist.cpu().detach().numpy(), label=f'action {action_id}')
                plt.legend()
                plt.pause(0.01)


        q = torch.sum(dist * self.support, dim=2) # a vector with the q value of each action, which is the expectation of the distribution shown in dist (q(s,a) = sum_i(dist[i] * support[i]).
        return q
    
    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        x = self.feature_layer(x)
        x = self.feature_layer2(x)
        x = self.feature_layer3(x)
        x = self.feature_layer4(x)
        x = self.feature_layer5(x)
        feature = self.feature_layer6(x)
        adv_hid = F.relu(self.advantage_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))
        
        advantage = self.advantage_layer(adv_hid).view(
            -1, self.out_dim, self.atom_size
        )
        value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)
        
        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans
        
        return dist
    
    def reset_noise(self):
        """Reset all noisy layers."""
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()