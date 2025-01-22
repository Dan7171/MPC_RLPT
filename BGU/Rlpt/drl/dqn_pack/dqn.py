"""
DQN implementation.
source: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#replay-memory
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F



class DQN(nn.Module):

    def __init__(self, state_dim, n_actions):
        
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(state_dim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 128)
        self.layer4 = nn.Linear(128, 128)
        self.layer5 = nn.Linear(128, 128)
        self.layer6 = nn.Linear(128, n_actions)
        # self.max_padded_dim = max(max_padded_dim, state_dim)
        
    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x) -> torch.Tensor:
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        return self.layer6(x)
    
    
    def pad_input(self, input_tensor):
        # Pad with zeros to max_length if needed
        padded_tensor = torch.zeros(1, self.max_padded_dim)
        padded_tensor[0, :input_tensor.shape[1]] = input_tensor # zeros on the right
        return padded_tensor

