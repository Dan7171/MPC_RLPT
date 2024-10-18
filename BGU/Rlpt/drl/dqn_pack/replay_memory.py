
"""
A replay buffer for DQN 
source: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#replay-memory:~:text=%22cpu%22%0A)-,Replay%20Memory,-We%E2%80%99ll%20be%20using

"""

import random
from collections import namedtuple, deque
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):
    """
    This class is the implementation of "replay buffer" performs "EXPERIENCE REPLAY": one of the tricks of this algorithm.
    The idea: To remove correlations, build dataset from agent’s own experience (maked samples (transitions: (st, at, rt+1, st+1) tuples) more iid)
 
    Args:
        object (_type_): _description_
    """

    def __init__(self, capacity, seed=-1):
        self.memory = deque([], maxlen=capacity)
        if seed != -1 and type(seed) == int:
            random.seed(seed)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        """
        Sample random mini-batch of transitions (s, a, r, s’) from D 
        """
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)