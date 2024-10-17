# import numpy as np
# import gymnasium as gym
# class Environment(gym.Env):
#     def __init__(self, actions):
        
#         # Observations are dictionaries with the agent's and the target's location.
#         # Each location is encoded as an element of {0, ..., `size`}^2, i.e. MultiDiscrete([size, size]).
#         self.observation_space = gym.spaces.Dict(
#             {
#                 "agent": (0, size - 1, shape=(2,), dtype=int),
#                 "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
#             }
#         )
        
#         self.action_space = gym.spaces.Discrete(len(actions))
