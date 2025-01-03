import copy
from importlib import metadata
import math
from os import access
from turtle import position
from typing import Dict, List, Set, Tuple, Union
from click import BadArgumentUsage
# from examples.franka_osc import orientation_error
from networkx import union
from storm_kit.mpc import task
import torch
from yaml import compose
# from BGU.Rlpt.drl.dqn_pack import train_suit
from BGU.Rlpt.DebugTools.globs import GLobalVars
from BGU.Rlpt.drl.dqn_pack.train_suit import trainSuit
from BGU.Rlpt.configs.default_main import load_config_with_defaults

import numpy as np
import pandas as pd
def get_obj_type(obj_properties):
    if 'radius' in obj_properties:
        return 'sphere'
    elif 'dims' in obj_properties:
        return 'cube'

def _merge_dict(original, update):
    new = copy.deepcopy(original)
    new.update(update)
    return new
    
class rlptAgent:       
    def __init__(self,base_pos_gym: np.ndarray, participating_storm:dict, not_participating_storm:dict,col_obj_handles:dict, action_space:list, max_col_objs=10):
        """
        Summary:
            initializng a reinforcement learning parameter tuning agent.
            
        Args:
            participating_storm (dict): Participating objects initial states {name:attributes}. Positions and orientations are in storm coordinate system.
            not_participating_storm (dict): Not participating objects initial states {name: attributes}. Positions and orientations are in storm coordinate system.
            col_obj_handles dict: a dict which keys are the names of collision objects (participating or not) and values are the object's handle (int)
            action_space (list): a collection of all possible actions which can be selected (all combinations of hyper parameters that can be selected at a single stime step)
        """
        # self.base_pos_gym_s0 = base_pos_gym
        # self.participating_storm:dict = participating_storm # initial states of participating collision objects. 
        # self.not_participating_storm:dict = not_participating_storm # initial states of collision objects which are not participating (). 
        self.action_space:list = action_space
        self.col_obj_handles:dict = col_obj_handles
        self.all_coll_obj_names_sorted:list = sorted(list(self.col_obj_handles.keys()), key=lambda x: self.col_obj_handles[x]) # sorted by obj handle
        # self.all_coll_objs_s0 = _merge_dict(participating_storm, not_participating_storm)
        all_coll_objs_s0 = _merge_dict(participating_storm, not_participating_storm)
        if len(all_coll_objs_s0) != len(participating_storm) + len(not_participating_storm):
            raise BadArgumentUsage("participating and non participating objects must contain objects with unique names") 
        # self.max_col_objs = max_col_objs        
        self.col_obj_s0_flat_states:dict[str,np.ndarray]= self.flatten_coll_obj_states(all_coll_objs_s0) # {'obj name': flattened objected state in storm cs([5,1,3,0.4...])}
        col_obj_s0_sorted_concat:np.ndarray = self.flatten_sorted_coll_objs_states(self.col_obj_s0_flat_states) # concatenated flattened objected states sorted by obj handle
        self.max_horizon = self._get_max_h() # maximun horizon in action space
        
        
        
        # all possible state representation components and their dimensions
        state_var_to_dim =  {
            'robot_dofs_positions': 7, # 1 scalar (angular position w.r to origin (0)) for each dof (joint) of the 7 dofs ,
            'robot_dofs_velocities': 7, # similarly to positions, an angular velocity on each dof 
            'goal_pose':  7, # position (3), orientation (4)
            'coll_objs': 7 * len(col_obj_s0_sorted_concat), # obj size (7) times num of collision objs
            'robot_base_pos': 3, # xyz (position only)
            'prev_action_idx': 1, # the index of the previous action which was taken
            'pi_mppi_means':  7 * self.max_horizon, # MPPI policy (H gaussians) means: 7 distribution means (one for each dof) for max- H actions
            'pi_mppi_covs': 7 # MPPI policy (H gaussians) covariances: 7 covariances of those means (unlike the means, the covs remain the same for the whole horizon)
        }
        
        # define the state representation configuration
        rlpt_cfg = GLobalVars.rlpt_cfg
        state_represantation_config = rlpt_cfg['agent']['model']['state_representation'] 
        self.st_componentes_ordered = []
        for var in state_represantation_config.keys():    
            if state_represantation_config[var]:
                self.st_componentes_ordered.append(var)
                
        # define the current state s(t) (initially s0)
        self.current_st = {component: np.array([]) for component in self.st_componentes_ordered} 
        if 'robot_base_pos' in self.st_componentes_ordered:
            self.current_st['robot_base_pos'] = base_pos_gym
        if 'coll_objs' in self.st_componentes_ordered:
            self.current_st['coll_objs'] = col_obj_s0_sorted_concat
        if 'prev_action_idx' in self.st_componentes_ordered:
            self.current_st['prev_action_idx'] = np.array([-1])
            
        
        
            
        # {'robot_base_pos': self.base_pos_gym_s0, 
        #       'robot_dofs_positions': robot_dof_positions_gym,
        #       'robot_dofs_velocities': robot_dof_velocities_gym, 
        #       'goal_pose': goal_pose_gym, # 
        #       'prev_action_idx': prev_at_idx_np,
        #       'coll_objs': self.col_obj_s0_sorted_concat,
        #       'pi_mppi_means': pi_mppi_means_padded,
        #       'pi_mppi_covs': pi_mppi_covs
        # }
        
        
        
        self.st_componentes_ordered_dims = [state_var_to_dim[component] for component in self.st_componentes_ordered]         
        self.st_dim:int = sum(self.st_componentes_ordered_dims) # len of each state s(t) (NN input length)
        self.st_legend = self.get_states_legend() # readable shape of the state 
        self.train_suit = trainSuit(self.st_dim , len(action_space)) # bulding the DQN (state dim is input dim,len action space is  output dim)
        self.shared_action_features, self.unique_action_features_by_idx = self.action_space_info() # mostly for printing
        
 
    
    def flatten_coll_obj_states(self, all_coll_objs_st):
        
        out: dict[str,np.ndarray] = {}
        for obj_name in self.all_coll_obj_names_sorted:
            nested_obj_state = list(all_coll_objs_st[obj_name].values()) 
            flattened_obj_state = np.concatenate([np.atleast_1d(x) for x in nested_obj_state]) # [x,[y],z,[t,w]]] -> [[x],[y],[z],[t,w]] -> [x,y,z,t,w]
            out[obj_name] = flattened_obj_state
        return out
    
    def parse_st(self,st:np.ndarray)-> Dict[str,np.ndarray]:    
        out = {}
        for tup in self.st_legend:
            component_range, component_name = tup
            start = component_range[0]
            stop = component_range[1]
            out[component_name] = st[start:stop+1]
        return out
    def print_state(self,st):
        
        st_parsed = self.parse_st(st)
        print('s(t):')
        for item_name, item_st  in st_parsed.items():
            if 'mppi' in item_name:
                print('skipped mppi policy printing')
                continue
            print(f'{item_name}: {item_st}')
            
    def select_action(self, st:torch.Tensor,forbidden_action_indices):
        """Given state s(t) return action a(t) and its index
        
        Args:
            st (torch.Tensor): s(t)

        Returns:
            _type_: _description_
        """
        action_idx_tensor: torch.Tensor
        action_idx:int
        action_idx_tensor, meta_data = self.train_suit.select_action_idx(st, forbidden_action_indices)
        action_idx = int(action_idx_tensor.item()) # action's index
        return action_idx, self.action_space[action_idx], meta_data # the action itself 
    
    def load(self, checkpoint):
        self.train_suit.current.load_state_dict(checkpoint['current_state_dict'])
        self.train_suit.target.load_state_dict(checkpoint['target_state_dict'])
        self.train_suit.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_suit.memory = checkpoint['memory']
        self.train_suit.steps_done = checkpoint['steps_done']
        
    def save(self, ep, ts, steps_done, model_file_path):
        # save model with relevant info to start the next episode
        torch.save({
            'current_state_dict': self.train_suit.current.state_dict(),
            'target_state_dict': self.train_suit.target.state_dict(),
            'optimizer_state_dict': self.train_suit.optimizer.state_dict(),
            'memory':self.train_suit.memory,
            'episode': ep,  # Optional: if you want to save the current epoch number
            'ts': ts,
            'steps_done': steps_done,
            
        }, model_file_path)
        
    def flatten_sorted_coll_objs_states(self, col_obj_st_flat_states):
        """ 
        Flatten collision object locs (participating and not) to ndarray of all objects states
        Returns:

        """
        out = np.array([]) # one long state, will contain sequenced coll objs states (participating and not) sorted by their handle
        for obj_name in self.all_coll_obj_names_sorted:
            out= np.append(out, col_obj_st_flat_states[obj_name]) # np.append([1, 2, 3], [[4, 5, 6], [7, 8, 9]]) - >array([1, 2, 3, ..., 7, 8, 9])
        return out
    
    def compose_state_vector(self, robot_dof_positions_gym: np.ndarray, robot_dof_velocities_gym:np.ndarray, goal_pose_gym:np.ndarray, pi_mppi_means: np.ndarray, pi_mppi_covs:np.ndarray, prev_at_idx=-1) -> np.ndarray:
        """ given components of state, return encoded (flatten) state

        Args:
            st (_type_): _description_
        """
        
        
        # Update the current state with the new components (only for components which can be changed)
        if 'prev_action_idx' in self.st_componentes_ordered:
            self.current_st['prev_action_idx'] = np.array([prev_at_idx]) 
        if 'pi_mppi_means' in self.st_componentes_ordered:
            self.current_st['pi_mppi_means'] = self._pad_pi_with_zeros(pi_mppi_means) # pi_mppi_means.flatten
        if 'pi_mppi_covs' in self.st_componentes_ordered:
            self.current_st['pi_mppi_covs'] = pi_mppi_covs # pi_mppi_means.flatten
        if 'robot_dofs_positions' in self.st_componentes_ordered:
            self.current_st['robot_dofs_positions'] = robot_dof_positions_gym
        if 'robot_dofs_velocities' in self.st_componentes_ordered:
            self.current_st['robot_dofs_velocities'] = robot_dof_velocities_gym
        if 'goal_pose' in self.st_componentes_ordered:
            self.current_st['goal_pose'] = goal_pose_gym
        
        # Vectorize the state components in the order of the state representation configuration
        ordered_st_components = [self.current_st[key].flatten() for key in self.st_componentes_ordered]
        current_st_vectorized = np.concatenate(ordered_st_components)    
        return current_st_vectorized
        
    
        
    
    def get_states_legend(self)->List[tuple]:
        
        out = []
        start_idx = 0
        for i in range(len(self.st_componentes_ordered)):
            if self.st_componentes_ordered[i] != 'coll_objs':
                component_name = self.st_componentes_ordered[i]
                component_len_in_st = self.st_componentes_ordered_dims[i]
                end_idx = start_idx + component_len_in_st - 1 # inclusive
                out.append(((start_idx, end_idx),component_name))
                print(component_name, (start_idx, end_idx))
                start_idx = end_idx + 1
            else:
                for j in range(len(self.all_coll_obj_names_sorted)):
                    component_name = self.all_coll_obj_names_sorted[j]
                    component_len_in_st = len(self.col_obj_s0_flat_states[component_name]) # obj shape
                    end_idx = start_idx + component_len_in_st - 1 # inclusive
                    out.append(((start_idx, end_idx),component_name))
                    print(component_name, (start_idx, end_idx))
                    start_idx = end_idx + 1
        return out
                    
                    
                
    def compute_reward(self, ee_pos_error, ee_rot_error, contact_detected, step_duration)->Tuple[np.float32, bool]:
        """ A weighted sum of reward terms
        Returns:
            a. np.float64: total reward of the transition from s(t) to s(t+1).
            b. is terminal state (contact_detected or goal_test)
        """
        rlpt_cfg = GLobalVars.rlpt_cfg
        reward_config = rlpt_cfg['agent']['reward']
        goal_pos_thresh_dist =  reward_config['goal_pos_thresh_dist']
        goal_rot_thresh_dist =  reward_config['goal_rot_thresh_dist']
        step_dur_w = reward_config['step_dur_w']
        safety_w = reward_config['safety_w']
        pose_w = reward_config['pose_w']

        assert_positive = [pose_w, safety_w, step_dur_w, goal_pos_thresh_dist,goal_rot_thresh_dist]
        for arg in assert_positive:
            arg_name = f'{arg=}'.split('=')[0]
            assert arg >= 0, BadArgumentUsage(f"argument {arg_name} must be positv, but {arg} was passed.")

        # checking if the goal is reached (if the ee is close enough to the goal position and orientation)
        goal_test = ee_pos_error < goal_pos_thresh_dist and ee_rot_error < goal_rot_thresh_dist
        
        if reward_config['pose_reward']:
            # positive rewards for position  and orientation when close enough to goal position        
            possition_reward =  max(0, goal_pos_thresh_dist - ee_rot_error) # "reversed relu" (greater when error is approaching 0, never negative)
            orientation_reward = max(0, goal_rot_thresh_dist - ee_rot_error) # "reversed relu" (greater when error is approaching 0,  never negative)
            
            # pose reward logic: we compute a positive reward for pose, only if both position and orientation are close enough to the goal
            # the pose reward is the sum of the position and orientation rewards,  
            pose_reward = (possition_reward + orientation_reward) * int(goal_test)         
            pose_reward *= pose_w
        else:
            pose_reward = 0   
            
        safety_reward = safety_w * - int(contact_detected)
        
        
        if  goal_test:
            step_duration_reward = 0 # no time penalty when goal is reached
        
        else:
            if reward_config['time_reward'] == 'linear':
                step_duration_reward = step_dur_w * - step_duration
            else: # binary, penalty of 1 for every step
                step_duration_reward = -1

        total_reward = pose_reward + safety_reward + step_duration_reward
        return np.float32(total_reward), contact_detected or goal_test

    def action_space_info(self):
        
        # deep copy and flattening of the action space 
        action_space_flatten = copy.deepcopy(self.action_space)
        for i in range(len(action_space_flatten)):
            action_space_flatten[i] = copy.deepcopy(action_space_flatten[i])
            action_space_flatten[i]['mpc_params'].update(action_space_flatten[i]['cost_weights'])
            action_space_flatten[i]['mpc_params']['goal_pose'] = tuple(action_space_flatten[i]['mpc_params']['goal_pose'])            
            action_space_flatten[i] = action_space_flatten[i]['mpc_params']
 
        df = pd.DataFrame.from_records(action_space_flatten) # rows = actions, columns = action dofs (features) 
        # Find columns where all values along rows are equal (action features that are the same over all actions)
        equal_columns = df.columns[df.nunique() == 1]
        df_equality = df[equal_columns][:1:] # equality between actions
 
        # Drop the columns with all equal values (more these action features to stay just with varying features)
        df_diffs = df.drop(equal_columns, axis=1) # differences between actions
 
        # parse output before returning it
        print("rlpt action space: shared action featueres")
        print(df_equality)
        print("rlpt: action space: unique action featueres")
        print(df_diffs)
        shared_params_all_actions:dict = df_equality.to_dict(orient='records')[0] # shared params between all actions
        different_params_by_action_inx:list = df_diffs.to_dict(orient='records') # l[i] a dict of the ith action, containing the unique assignment of this action to action features with 2 or more options  
        return shared_params_all_actions, different_params_by_action_inx 
    
    def _pad_pi_with_zeros(self, mppi_pi_means_cur_h:np.ndarray) -> np.ndarray:
        """ adding (max H - curr H) steps to policy with action = [0,0,0,0,0,0,0] (zero at each dof) for padding 

        Args:
            mppi_pi_means_cur_h (np.ndarray): 

        Returns:
            _type_: _description_
        """
        
        cur_pi_horizon = mppi_pi_means_cur_h.shape[0]
        rows_to_pad = self.max_horizon - cur_pi_horizon
        result = mppi_pi_means_cur_h
        if rows_to_pad > 0: # current policy is for H smaller than max H 
            trailing_zeros = np.zeros((rows_to_pad, 7))
            result = np.vstack((mppi_pi_means_cur_h, trailing_zeros))
        else:
            assert rows_to_pad == 0, 'bug'
        
        return result
           
    def _get_max_h(self):
        cur_max = 0
        for action in self.action_space:
            cur_max = max(cur_max, action['mpc_params']['horizon'])
        return cur_max
        
  
            