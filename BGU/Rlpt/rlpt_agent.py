import torch
from BGU.Rlpt.drl.dqn_pack.train_suit import trainSuit

class rlptAgent:
    def __init__(self, state_dim_flatten, action_space):
        pass
        self.action_space = action_space
        self.state_dim_flatten = state_dim_flatten
        self.train_suit = trainSuit(state_dim_flatten , len(action_space))
    
    def select_action(self, st):
        action_idx_tens: torch.Tensor = self.train_suit.select_action(st)
        action_idx = action_idx_tens.item() # a index
        return self.action_space[action_idx] # a

    def encode_state(self, st):
        """return encoded state

        Args:
            st (_type_): _description_
        """
        pass
    
    def compute_reward(self, ee_pos_error, ee_rot_error, primitive_collision_error, step_duration):
        
        alpha, beta, gamma, delta = 1, 1, 1, 1
        
        pose_error = alpha * ee_pos_error + beta * (ee_rot_error / ee_pos_error)
        pose_reward = - pose_error
        
        primitive_collision_reward = gamma * primitive_collision_error
        
        step_duration_reward = delta * -step_duration
        
        total_reward = pose_reward + primitive_collision_reward + step_duration_reward
        
        return total_reward
        
        
  
            