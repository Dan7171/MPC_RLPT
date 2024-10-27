import copy
from importlib import metadata
import math
from os import access
from typing import Dict, List, Set, Tuple, Union
from click import BadArgumentUsage
from networkx import union
from storm_kit.mpc import task
import torch
from BGU.Rlpt.drl.dqn_pack.train_suit import trainSuit
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
    # NO_OP_CODE = 'no_op' # A constant representing a special action which every rlpt agent has. It's meaning is doing nothing. So if a(t) is NO_OP: rlpt won't tune parameters in time t (and system will be based on previous parameters)
    
 
        
    def __init__(self,base_pos_gym: np.ndarray, participating_storm:dict, not_participating_storm:dict,col_obj_handles:dict, action_space:list):
        """
        Summary:
            initializng a reinforcement learning parameter tuning agent.
            
        Args:
            participating_storm (dict): Participating objects initial states {name:attributes}. Positions and orientations are in storm coordinate system.
            not_participating_storm (dict): Not participating objects initial states {name: attributes}. Positions and orientations are in storm coordinate system.
            col_obj_handles dict: a dict which keys are the names of collision objects (participating or not) and values are the object's handle (int)
            action_space (list): a collection of all possible actions which can be selected (all combinations of hyper parameters that can be selected at a single stime step)
        """
        self.base_pos_gym_s0 = base_pos_gym
        self.participating_storm:dict = participating_storm # initial states of participating collision objects. 
        self.not_participating_storm:dict = not_participating_storm # initial states of collision objects which are not participating (). 
        self.action_space:list = action_space
        self.col_obj_handles:dict = col_obj_handles
        self.all_coll_obj_names_sorted:list = sorted(list(self.col_obj_handles.keys()), key=lambda x: self.col_obj_handles[x]) # sorted by obj handle
        self.all_coll_objs_s0 = _merge_dict(self.participating_storm, self.not_participating_storm)
        if len(self.all_coll_objs_s0) != len(self.participating_storm) + len(self.not_participating_storm):
            raise BadArgumentUsage("participating and non participating objects must contain objects with unique names") 
        self.col_obj_s0_flat_states:dict[str,np.ndarray]= self.flatten_coll_obj_states(self.all_coll_objs_s0) # {'obj name': flattened objected state in storm cs([5,1,3,0.4...])}
        self.col_obj_s0_sorted_concat:np.ndarray = self.flatten_sorted_coll_objs_states(self.col_obj_s0_flat_states) # concatenated flattened objected states sorted by obj handle
        
        # define every s(t) specifications
        self.st_componentes_ordered = ['robot_base_pos',
                                       'robot_dofs_positions',
                                       'robot_dofs_velocities', 
                                       'goal_pose',  
                                       'prev_action_idx',
                                       'coll_objs'
                                       ]
        
        self.st_componentes_ordered_dims = [3, # x,y,z pos of robot base (3)
                                            7, # 1 scalar (angular position w.r to origin (0)) for each dof (joint) of the 7 dofs 
                                            7, # similarly to positions, an angular velocity on each dof 
                                            7, # position (3), orientation (4)
                                            1, # the index of the previous action which was taken
                                            len(self.col_obj_s0_sorted_concat) # flatten state length (NN input length). 
                                            ]  
        self.st_dim:int = sum(self.st_componentes_ordered_dims) # len of each state s(t)
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
    
    def compose_state_vector(self, robot_dof_positions_gym: np.ndarray, robot_dof_velocities_gym:np.ndarray, goal_pose_gym:np.ndarray, prev_at_idx:Union[None, int]) -> np.ndarray:
        """ given components of state, return encoded (flatten) state

        Args:
            st (_type_): _description_
        """
        no_prev_action_code = -1
        if prev_at_idx is None:
            prev_at_idx = no_prev_action_code
        prev_at_idx_np = np.array([prev_at_idx]) # a special code to represent n meaning

        st = {'robot_base_pos': self.base_pos_gym_s0, 
              'robot_dofs_positions': robot_dof_positions_gym,
              'robot_dofs_velocities': robot_dof_velocities_gym, 
              'goal_pose': goal_pose_gym, # 
              'prev_action_idx': prev_at_idx_np,
              'coll_objs': self.col_obj_s0_sorted_concat
        }
        ordered_st_components = [st[key] for key in self.st_componentes_ordered]
        return np.concatenate(ordered_st_components)
    
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
                    
                    
                
    def compute_reward(self, ee_pos_error, ee_rot_error, contact_detected, step_duration, pos_w=1, col_w=5000, step_dur_w=50, pos_r_radius=0.1, pos_r_sharpness=50)->np.float64:
        """ A weighted sum of reward terms considering next terms:
            1. ee_pos_error: position distance from goal (ee_pos_error, l2 norm of the difference between current ee pos and goal ee pos), 
            2. ee_rot_error: orientation distance" (l2 norm of the difference between current and goal)
            # 3. primitive_collision_error: error taken from storm system, Correlated with the distance from obstacles. 
            4. step_duration - the time it took to execute transition s(t) to s(t+1)
        Args:
            ee_pos_error (float): l2 norm of the distance to target error (position error): {v = (x,y,z) = (current end effector location - goal end effector location)} of the transition from s(t) to s(t+1) 
            ee_rot_error (float): l2 norm of the quaternion error (rotation error): {v = (r1,r2,r3,r4) = (current end effector rotation - goal end effector rotation)} of the transition from s(t) to s(t+1) 
            # primitive_collision_error (float): premitive ("with obstacles") collision error of the transition from s(t) to s(t+1)  
            step_duration (float): the time it took to tune the params (perform action a(t) of rlpt) + the time it took to execute the step in robot
            pos_w (int, optional): Defaults to 1. the weight of position reward in total reward calculation of the transition..
            col_w (int, optional): Defaults to 1.the weight of premitive ("with obstacles") collision reward in total reward calculation of the transition.
            step_dur_w (int, optional):. Defaults to 1.  the weight of step duration reward in total reward calculation of the transition.
            
            pos_r_radius (float, optional): Defaults to 0.05. "Radius of interest" of position error in centimeters. It affects position reward- the greater it is, the further from goal position (earlier in route) that the arm starts being aware to position error.  
                This is the exponent sharpness. the greater it is, the sharper/faster the orientation reward weight (how much we care about orientation) is inceased when we cross the radius towards the exact goal position. 
                Position reward logic:  
                1. When ee position is at distance pos_r_radius (in cm) from goal- the reward is 1. If distance from goal is infinity, the reward is 0
                2. The smaller the distance to goal gets, reward climbs
                3. At distance is pos_r_radius ()"the radius of interest") or smaller, the reward starts to climb sharply. In other words, pos_r_radius represents 
                a sphere in that shape around the goal position, which when the ee is entering that sphere, you start treating that entry as reaching close enough to the exact goal pose (and if the ee is outside that sphere, you don't really care or consider it as sucecss)).
                Meaning: the smaller you set pos_r_radius, the more precise you want ee to reach, and you consider a smaller sphere around goal positio as success.
                4. The greater pos_r_sharpness is, the faster the reward climbs when you get close to the goal position (it mostly gets when you cross the "pos_r_radius")
            pos_r_sharpness (int, optional): Defaults to 50. the weight of position reward in total reward calculation of the transition..
            
        Returns:
            np.float64: total reward of the transition from s(t) to s(t+1).
        """
        # pos_eps_ee_convergence = 0.01
        # rot_eps_ee_convergence = 0.01
        
        
        assert_positive = [pos_w, col_w, step_dur_w, pos_r_radius, pos_r_sharpness]
        for arg in assert_positive:
            arg_name = f'{arg=}'.split('=')[0]
            assert arg > 0, BadArgumentUsage(f"argument {arg_name} must be positv, but {arg} was passed.")
         
        postion_reward = math.exp(pos_r_sharpness *(-ee_pos_error + pos_r_radius)) # e^(sharp*(-pos_err + rad))
        orient_w = postion_reward # We use the position reward as the weight for the orientation reward, as we want the orientation reward to be more significant (either good or bad) as we get closer to goal in terms of position.    
        orientation_reward = orient_w * - ee_rot_error  
        primitive_collision_reward = col_w * - int(contact_detected)
        step_duration_reward = step_dur_w * - step_duration
        total_reward = postion_reward + orientation_reward + primitive_collision_reward + step_duration_reward
        
        
        # print(f"rewards: position, orientation, premitive-collision , step duration\n{postion_reward:{.3}f}, {orientation_reward:{.3}f}, {primitive_collision_reward:{.3}f}, {step_duration_reward:{.3}f}")
#         print(f"r(t) (term, reward):\n\
# position (ee to goal distance, r), orientation (ee to goal distance, r), prim-coll (error, r), step duration (duration, r)\n({ee_pos_error:{.3}f}, {postion_reward:{.3}f}), ({ee_rot_error:{.3}f}, {orientation_reward:{.3}f}), ({primitive_collision_error:{.3}f},{primitive_collision_reward:{.3}f}),({step_duration:{.3}f}, {step_duration_reward:{.3}f})")
        print(f"r(t) (term, reward):\n\
position (ee to goal distance, r), orientation (ee to goal distance, r), prim-coll (was contact, r), step duration (duration, r)\n({ee_pos_error:{.3}f}, {postion_reward:{.3}f}), ({ee_rot_error:{.3}f}, {orientation_reward:{.3}f}), ({contact_detected}, {primitive_collision_reward:{.3}f}),({step_duration:{.3}f}, {step_duration_reward:{.3}f})")
        
        return total_reward

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
    
                
                
        
  
            