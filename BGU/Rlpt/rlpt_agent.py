import copy
from typing import List, Set
from click import BadArgumentUsage
from networkx import union
from storm_kit.mpc import task
import torch
from BGU.Rlpt.drl.dqn_pack.train_suit import trainSuit
import numpy as np

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
    NO_OP_CODE = 'no_op' # A constant representing a special action which every rlpt agent has. It's meaning is doing nothing. So if a(t) is NO_OP: rlpt won't tune parameters in time t (and system will be based on previous parameters)
    
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
        self.train_suit = trainSuit(self.state_dim , len(action_space)) # input dim, output di,m
        
        
    def _calc_state_dimension(self) -> int:
        
        # robot_base_pos_dim = 3 # x,y,z
        robot_dofs_positions_dim = 7 # 1 scalar (angular position w.r to origin (0)) for each dof (joint) of the 7 dofs 
        robot_dofs_velocities_dim = 7 # an angular velocity on each dof 
        goal_pose_dim = 7 # position (3), orientation (4)
        
        task_section_size = goal_pose_dim  # section size from state
        robot_section_size = robot_dofs_positions_dim + robot_dofs_velocities_dim # section size from state
        
        # objectes_section_size = sphere_dim * n_spheres + cube_dim * n_cubes # section size from state 
        objects_section_size = len(self.all_coll_objs_initial_state)
        return robot_section_size + task_section_size + objects_section_size
    
    def select_action(self, st:torch.Tensor):
        """Given state s(t) return action a(t)

        Args:
            st (_type_): _description_

        Returns:
            _type_: _description_
        """
        action_idx_tensor: torch.Tensor
        action_idx:int
        
        action_idx_tensor = self.train_suit.select_action(st)
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
    
    def compose_state_vector(self, robot_dof_positions_gym: np.ndarray, robot_dof_velocities_gym:np.ndarray, goal_pose_gym:np.ndarray) -> np.ndarray:
        """ given components of state, return encoded (flatten) state

        Args:
            st (_type_): _description_
        """
        # return np.ndarray([self.all_coll_objs_initial_state,robot_dof_positions, robot_dof_velocities, goal_pose]).ravel()
        return np.concatenate([self.all_coll_objs_initial_state,robot_dof_positions_gym, robot_dof_velocities_gym, goal_pose_gym])
    
    def compute_reward(self, ee_pos_error, ee_rot_error, primitive_collision_error, step_duration)->np.float64:
        
        alpha, beta, gamma, delta = 1, 1, 1, 1
        
        pose_error = alpha * ee_pos_error + beta * (ee_rot_error / ee_pos_error)
        pose_reward = - pose_error
        
        primitive_collision_reward = - gamma * primitive_collision_error
        
        step_duration_reward = delta * - step_duration
        
        total_reward = pose_reward + primitive_collision_reward + step_duration_reward
        
        print(f"pose_reward,  primitive_collision_reward,  step_duration_reward\n\
              {pose_reward}, {primitive_collision_reward}, {step_duration_reward}")
        return total_reward
        
        
  
            