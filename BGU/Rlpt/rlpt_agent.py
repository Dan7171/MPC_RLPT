import copy
import math
from os import access
from typing import List, Set, Union
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
    
 
        
    def __init__(self, participating_storm:dict, not_participating_storm:dict,col_obj_handles:dict, action_space:list):
        """
        Summary:
            initializng a reinforcement learning parameter tuning agent.
            
        Args:
            participating_storm (dict): Participating objects initial states {name:attributes}. Positions and orientations are in storm coordinate system.
            not_participating_storm (dict): Not participating objects initial states {name: attributes}. Positions and orientations are in storm coordinate system.
            col_obj_handles dict: a dict which keys are the names of collision objects (participating or not) and values are the object's handle (int)
            action_space (list): a collection of all possible actions which can be selected (all combinations of hyper parameters that can be selected at a single stime step)
        """
        
        self.participating_storm:dict = participating_storm # initial states of participating collision objects. 
        self.not_participating_storm:dict = not_participating_storm # initial states of collision objects which are not participating (). 
        self.action_space:list = action_space
        self.col_obj_handles:dict = col_obj_handles
        self.all_coll_obj_names_sorted:list = sorted(list(self.col_obj_handles.keys()), key=lambda x: self.col_obj_handles[x]) # sorted by obj handle
        self.all_objs = _merge_dict(self.participating_storm, self.not_participating_storm)
        if len(self.all_objs) != len(self.participating_storm) + len(self.not_participating_storm):
            raise BadArgumentUsage("participating and non participating objects must contain objects with unique names") 
        self.all_coll_objs_initial_state:np.ndarray = self._parse_coll_objs_state() # in storm cs
        self.state_dim = self._calc_state_dimension() # input dimension for the ddqn. 
        self.train_suit = trainSuit(self.state_dim , len(action_space)) # input dim, output dim
        self.shared_action_features, self.unique_action_features_by_idx = self.action_space_info()
        
     
    def _calc_state_dimension(self) -> int:
        
        # robot_base_pos_dim = 3 # x,y,z
        robot_dofs_positions_dim = 7 # 1 scalar (angular position w.r to origin (0)) for each dof (joint) of the 7 dofs 
        robot_dofs_velocities_dim = 7 # an angular velocity on each dof 
        goal_pose_dim = 7 # position (3), orientation (4)
        prev_action_idx_dim = 1 # the index of the previous action which was taken
        # steps_from_horizon_switch_dim = 1 # the number of steps passed from the last action where we actually switched the horizon

        
        task_section_size = goal_pose_dim  # section size from state
        robot_section_size = robot_dofs_positions_dim + robot_dofs_velocities_dim # section size from state
        # objectes_section_size = sphere_dim * n_spheres + cube_dim * n_cubes # section size from state 
        objects_section_size = len(self.all_coll_objs_initial_state)
        prev_action_idx_section_size = prev_action_idx_dim
        # h_switch_counter_section_size = steps_from_horizon_switch_dim
        return robot_section_size + task_section_size + objects_section_size + prev_action_idx_section_size
    
        # return robot_section_size + task_section_size + objects_section_size + prev_action_idx_section_size + h_switch_counter_section_size
        
    def select_action(self, st:torch.Tensor,forbidden_action_indices):
        """Given state s(t) return action a(t) and its index
        
        Args:
            st (torch.Tensor): s(t)

        Returns:
            _type_: _description_
        """
        action_idx_tensor: torch.Tensor
        action_idx:int
        
        action_idx_tensor = self.train_suit.select_action_idx(st, forbidden_action_indices)
        action_idx = int(action_idx_tensor.item()) # action's index
        return action_idx, self.action_space[action_idx] # the action itself 
    
    
    
    def _parse_coll_objs_state(self):
        """ 
        Flatten collision object locs (participating and not) to ndarray of all objects states
        Returns:

        """
        
        all_objs_state_flatten = np.array([]) # one long state, will contain sequenced coll objs states (participating and not) sorted by their handle
        for obj_name in self.all_coll_obj_names_sorted:
            nested_obj_state = list(self.all_objs[obj_name].values()) 
            flattened_obj_state = np.concatenate([np.atleast_1d(x) for x in nested_obj_state]) # [x,[y],z,[t,w]]] -> [[x],[y],[z],[t,w]] -> [x,y,z,t,w]
            all_objs_state_flatten = np.append(all_objs_state_flatten, flattened_obj_state) # np.append([1, 2, 3], [[4, 5, 6], [7, 8, 9]]) - >array([1, 2, 3, ..., 7, 8, 9])
        return all_objs_state_flatten
    
    def compose_state_vector(self, robot_dof_positions_gym: np.ndarray, robot_dof_velocities_gym:np.ndarray, goal_pose_gym:np.ndarray, prev_at_idx:Union[None, int]) -> np.ndarray:
        """ given components of state, return encoded (flatten) state

        Args:
            st (_type_): _description_
        """
        no_prev_action_code = -1
        if prev_at_idx is None:
            prev_at_idx = no_prev_action_code
        prev_at_idx_np = np.array([prev_at_idx]) # a special code to represent n meaning
        return np.concatenate([self.all_coll_objs_initial_state,robot_dof_positions_gym, robot_dof_velocities_gym, goal_pose_gym, prev_at_idx_np])
    
    def compute_reward(self, ee_pos_error, ee_rot_error, contact_detected, step_duration, pos_w=1, col_w=5000, step_dur_w=1, pos_r_radius=0.1, pos_r_sharpness=50)->np.float64:
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
        pos_eps_ee_convergence = 0.01
        rot_eps_ee_convergence = 0.01
        
        
        assert_positive = [pos_w, col_w, step_dur_w, pos_r_radius, pos_r_sharpness]
        for arg in assert_positive:
            arg_name = f'{arg=}'.split('=')[0]
            assert arg > 0, BadArgumentUsage(f"argument {arg_name} must be positv, but {arg} was passed.")
         
        postion_reward = math.exp(pos_r_sharpness *(-ee_pos_error + pos_r_radius)) # e^(sharp*(-pos_err + rad))
        orient_w = postion_reward # We use the position reward as the weight for the orientation reward, as we want the orientation reward to be more significant (either good or bad) as we get closer to goal in terms of position.    
        orientation_reward = orient_w * - ee_rot_error  
        # primitive_collision_reward = col_w * - primitive_collision_error        
        primitive_collision_reward = col_w * - int(contact_detected)
        if ee_pos_error < pos_eps_ee_convergence and ee_rot_error < rot_eps_ee_convergence: #if in target
            print(f"'\033[94m'In convergence zone'\033[0m'") # blue
            step_dur_w = 0 # don't punish on step duration to target since we are alreay in target
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
    
                
                
        
  
            