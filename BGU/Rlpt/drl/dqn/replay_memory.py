
"""
A replay buffer for DQN 
source: https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html#replay-memory:~:text=%22cpu%22%0A)-,Replay%20Memory,-We%E2%80%99ll%20be%20using

"""
import random
from collections import namedtuple, deque
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)