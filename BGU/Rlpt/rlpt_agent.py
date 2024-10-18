from storm_kit.mpc import task
import torch
from BGU.Rlpt.drl.dqn_pack.train_suit import trainSuit
import numpy as np

def get_obj_type(obj_properties):
    if 'radius' in obj_properties:
        return 'sphere'
    elif 'dims' in obj_properties:
        return 'cube'
    
    
class rlptAgent:
    def __init__(self, participating_coll_objs, not_participating_coll_objs, action_space):
        
        # robot section size in state
        # robot_base_pos_dim = 3 # x,y,z
        robot_dofs_positions_dim = 7 # 1 scalar (angular position w.r to origin (0)) for each dof (joint) of the 7 dofs 
        robot_dofs_velocities_dim = 7 # an angular velocity on each dof 
        robot_section_size = robot_dofs_positions_dim + robot_dofs_velocities_dim 
        
        # task section size in state
        goal_pose_dim = 7 # position (3), orientation (4)
        task_section_size = goal_pose_dim
      
        # objects section size in state
        
        sphere_dim = 4 # position of center (3), radius (1)
        cube_dim = 10  # position (3), orientation (4), width (1), height (1), depth (1) 
        
        
        all_spheres = {}
        all_cubes = {}
        
        for coll_objs in [participating_coll_objs, not_participating_coll_objs]:
            for obj_name in coll_objs:
                obj_type = get_obj_type(participating_coll_objs[obj_name]) 
                if obj_type == 'sphere':
                    d = all_spheres
                elif obj_type == 'cube':
                    d = all_cubes
                d[obj_name] = participating_coll_objs[obj_name]
        
        n_spheres = len(all_spheres) # all objects in file
        n_cubes = len(all_cubes) # all objects in file
        
        # n_cubes = len(mpc.get_actor_group_from_env('cube')) # all objects in file
        objectes_section_size = sphere_dim * n_spheres + cube_dim * n_cubes 
        
        # finally 
        # rlpt_state_dim = robot_section_size + objectes_section_size + task_section_size
        
        self.participating_coll_objs = participating_coll_objs # initial states. Will not change locs
        self.not_participating_coll_objs = participating_coll_objs # initial states. Will not change locs
        self.action_space = action_space
        self.flatten_obj_states:np.ndarray = self._parse_coll_objs_state()
        
        self.rlpt_state_dim = robot_section_size + task_section_size + objectes_section_size
        self.train_suit = trainSuit(self.rlpt_state_dim , len(action_space)) # input dim, output di,m
        
        
    def select_action(self, st):
        action_idx_tens: torch.Tensor = self.train_suit.select_action(st)
        action_idx = action_idx_tens.item() # a index
        return self.action_space[action_idx] # a
    
    
    
    def _parse_coll_objs_state(self):
        """ Flatten collision object locs (participating and not) to ndarray of all objects states

        Returns:

        """
        objs_state = np.array([])
        for coll_objs in [self.participating_coll_objs, self.not_participating_coll_objs]:
            for obj_name in coll_objs:
                nested_obj_state = list(coll_objs[obj_name].values()) 
                flattened_obj_state = np.concatenate([np.atleast_1d(x) for x in nested_obj_state]) # [x,[y],z,[t,w]]] -> [[x],[y],[z],[t,w]] -> [x,y,z,t,w]
                np.append(objs_state, flattened_obj_state) # np.append([1, 2, 3], [[4, 5, 6], [7, 8, 9]]) - >array([1, 2, 3, ..., 7, 8, 9])
        return objs_state
    
    def compose_state_vector(self, robot_dof_positions: np.ndarray, robot_dof_velocities:np.ndarray, goal_pose:np.ndarray):
        """ given components of state, return encoded (flatten) state

        Args:
            st (_type_): _description_
        """
        return np.ndarray([self.flatten_obj_states,robot_dof_positions, robot_dof_velocities, goal_pose]).ravel()
    
    def compute_reward(self, ee_pos_error, ee_rot_error, primitive_collision_error, step_duration):
        
        alpha, beta, gamma, delta = 1, 1, 1, 1
        
        pose_error = alpha * ee_pos_error + beta * (ee_rot_error / ee_pos_error)
        pose_reward = - pose_error
        
        primitive_collision_reward = gamma * primitive_collision_error
        
        step_duration_reward = delta * -step_duration
        
        total_reward = pose_reward + primitive_collision_reward + step_duration_reward
        
        return total_reward
        
        
  
            