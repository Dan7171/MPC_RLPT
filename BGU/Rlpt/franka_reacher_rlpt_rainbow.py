from flask.cli import F
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch

import copy
import datetime
import gc
from json import load
from operator import xor
from os import name
import os
import pickle
from re import I, match
from select import select
import shutil
from threading import Thread
from typing import Callable, Collection, Iterable, List, Tuple, Union
import typing
from click import BadArgumentUsage
from BGU.Rlpt.drl.rainbow_rlpt.dqn_agent import DQNAgent
from colorlog import root
from cv2 import norm
from matplotlib.transforms import Transform
import psutil
from pygame import Vector3
import scipy as sp
from storm_kit import mpc
from storm_kit.mpc.control import mppi
from storm_kit.mpc.cost import cost_base
from sympy import Integer, im
import torch
from traitlets import default
from BGU.Rlpt.rlpt_agent import rlptAgent
from BGU.Rlpt.rlpt_agent_base import rlptAgentBase 
torch.multiprocessing.set_start_method('spawn',force=True)
torch.set_num_threads(8)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import time
import yaml
import argparse
import numpy as np
from quaternion import quaternion, from_rotation_vector, from_rotation_matrix
import matplotlib.pyplot as plt
import random
import math
from scipy.spatial.transform import Rotation as R
from quaternion import from_euler_angles, as_float_array, as_rotation_matrix, from_float_array, as_quat_array
from storm_kit.gym.core import Gym, World
from storm_kit.gym.sim_robot import RobotSim
from storm_kit.util_file import get_configs_path, get_gym_configs_path, join_path, load_yaml, get_assets_path
from storm_kit.gym.helpers import load_struct_from_dict
from storm_kit.util_file import get_mpc_configs_path as mpc_configs_path
from storm_kit.differentiable_robot_model.coordinate_transform import quaternion_to_matrix, CoordinateTransform
from storm_kit.mpc.task.reacher_task import ReacherTask
from BGU.Rlpt.DebugTools.CostFnSniffer import CostFnSniffer
from BGU.Rlpt.DebugTools.globs import GLobalVars
from BGU.Rlpt.configs.default_main import load_config_with_defaults
from BGU.Rlpt.configs.storm_original_params import mppi_space_default, cost_fn_space_default
import matplotlib.pyplot as plt
from BGU.Rlpt.experiments.experiment_utils import get_combinations
from deepdiff import DeepDiff
from multiprocessing import Process
import csv
from BGU.Rlpt.utils.type_operations import torch_tensor_to_ndarray, as_2d_tensor, as_1d_tensor
from BGU.Rlpt.utils.utils import make_model_path, color_print, print_progress_bar, goal_test
from BGU.Rlpt.utils.error import pos_as_ndarray, pos_error, rot_error, pose_as_ndarray
import psutil
import os
from BGU.Rlpt.rlpt_agent_base import META_DATA_SIGNATURE_ETL

np.set_printoptions(precision=2)
GREEN = gymapi.Vec3(0.0, 0.8, 0.0)
RED = gymapi.Vec3(0.8, 0.1, 0.1)
ONE_CM = 0.01 # IN METERS
now = time.time()
model_dir = os.path.join('BGU/Rlpt/trained_models') 


def get_actor_name(gym, env, actor_handle):
    return gym.get_actor_name(env, actor_handle)



def gui_draw_lines(gym_instance,mpc_control,w_robot_coord):
    """
    Drawing the green (good) & red (bed) trajectories in gui, at every real-world time step
    """
    # >>>>> Dan - removing the red and green colors from screen. Comment out to see what happens >>>>>>>>>>
    gym_instance.clear_lines()
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    top_trajs = mpc_control.top_trajs.cpu().float()#.numpy()
    n_p, n_t = top_trajs.shape[0], top_trajs.shape[1]
    w_pts = w_robot_coord.transform_point(top_trajs.view(n_p * n_t, 3)).view(n_p, n_t, 3)

    # >>>>> Dan - this block is making the green/red lines (good/bad trajectories) in gui at every step in gui. comment it out to see >>> 
    top_trajs = w_pts.cpu().numpy()
    color = np.array([0.0, 1.0, 0.0])
    for k in range(top_trajs.shape[0]):
        pts = top_trajs[k,:,:]
        color[0] = float(k) / float(top_trajs.shape[0])
        color[1] = 1.0 - float(k) / float(top_trajs.shape[0])
        gym_instance.draw_lines(pts, color=color)
    
# GPU usage profiling functions
def start_mem_profiling():
    """
    start profiling gpu memory consumption.
    ref: https://pytorch.org/docs/stable/torch_cuda_memory.html
    """
    torch.cuda.memory._record_memory_history(max_entries=100000,stacks='all')
def finish_mem_profiling(output_path):
    """
    stop profiling memory and save it to output path
    drag and drop output file here to analyze: https://pytorch.org/memory_viz
    """
    
    torch.cuda.memory._dump_snapshot(output_path)    
    print(f"memory usage profile was saved to {output_path}")
    

# GENERAL HELPER FUNCS
def reverse_map(map:dict):    
    ans = {}
    for k,v in map.items():
        ans[v] = k
    return ans

def make_rlpt_actionspace():
    # define rlpt action space
    
    if rlpt_cfg['agent']['action_space']['use_original_storm_params']:
        cost_fn_space = cost_fn_space_default
        mppi_space = mppi_space_default    
    
    else: 
        action_space_raw = rlpt_cfg['agent']['A']
        for action_space in action_space_raw.values():
            for dof_name, dof_options in action_space.items():
                for i in range(len(dof_options)):
                    if type(dof_options[i]) == list:
                        dof_options[i] = tuple(dof_options[i])
        cost_fn_space = action_space_raw['cost_fn_space']
        mppi_space = action_space_raw['mppi_space']    
                
    # This op is aimed to save time when some parameters like horizon are too expansive to modify 
    rlpt_action_space:list = list(get_combinations({
        'cost_weights': get_combinations(cost_fn_space),
        'mpc_params': get_combinations(mppi_space)}))

    return rlpt_action_space
class MpcRobotInteractive:
    """
    This class is for controlling the arm base and simulator.
    It contains the functions for RL learning.
    Operations to control the simulation
    TODO: Don't really like this class. This class should be called "Controller" or "Actions for simulation" since thats what it is
    """
    initial_ee_pose_storm_cs = np.array([]) # initial end effector pose in storm coordinate system

    def __init__(self, args, gym_instance: Gym, rlpt_cfg:dict, env_yml_relative_path:str,task_yml_relative_path:str):
        """ 

        Args:
            args (_type_): _description_
            gym_instance (_type_): 
            rlpt_cfg (dict): new parameters. Added to simplify code, improve readibility and more. 
        """
        
        self.gui_settings = rlpt_cfg['gui']    
        self.args = args
        self.gym_instance = gym_instance
        robot_name: str = self.args.robot # franka
        # RL variables
        self.ee_pose_storm_cs = (np.zeros(3) ,np.zeros(4)) # position, quaternion
        self.goal_pose = [0,0,0,0,0,0,1]
        self.benchmark_states = []
        
        # self.arm_configuration = None
        self.objects_configuration = None
        self.vis_ee_target = self.gui_settings['render_ee_icons']   # Display "red cup" (the end goal state/effector target location) in gui. Not effecting algorithm (navigation to target), just its representation in gui.
        # config files 
        self.robot_file = robot_name + '.yml' # collision spheres settings of the robot for each link (center, radius)
        self.task_file =  robot_name + '_reacher.yml' # task settings:  robot urdf, cost params, mpc (mppi) params and more 
        # self.world_file = 'collision_primitives_3d_origina2.yml' # world settings: assets in the world (except the franka)
        self.env_yml_relative_path = env_yml_relative_path
        # Simulator variables
        # self.pose = None # Goal pose in simulation
        self.gym = self.gym_instance.gym
        self.sim = self.gym_instance.sim
        with open(join_path(get_gym_configs_path(), self.env_yml_relative_path)) as file:
            self.world_params = yaml.load(file, Loader=yaml.FullLoader)
        
        self.robot_yml = join_path(get_gym_configs_path(), self.robot_file)
        with open(self.robot_yml) as file:
            self.robot_params = yaml.load(file, Loader=yaml.FullLoader)
        self.sim_params = self.robot_params['sim_params']
        self.sim_params['asset_root'] = get_assets_path()
        
        if(self.args.cuda):
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        self.sim_params['collision_model'] = None
        # create robot simulation:
        self.robot_sim = RobotSim(gym_instance=self.gym, sim_instance=self.sim, **self.sim_params, device=self.device)

        # create gym environment:
        self.robot_pose = self.sim_params['robot_pose']
        self.env_ptr = self.gym_instance.env_list[0] # first env (out of 1. one in total)
        self.robot_ptr = self.robot_sim.spawn_robot(self.env_ptr, self.robot_pose, coll_id=2) # robot
        
        # torch & cuda args
        self.device = torch.device('cuda', 0) 
        self.tensor_args = {'device':self.device, 'dtype':torch.float32}
    
        # spawn camera:
        self.robot_camera_pose = np.array([1.6,-1.5, 1.8,0.707,0.0,0.0,0.707])
        self.q = as_float_array(from_euler_angles(-0.5 * 90.0 * 0.01745, 50.0 * 0.01745, 90 * 0.01745))
        self.robot_camera_pose[3:] = np.array([self.q[1], self.q[2], self.q[3], self.q[0]])
        self.robot_sim.spawn_camera(self.env_ptr, 60, 640, 480, self.robot_camera_pose) 

        # get pose
 
        self.w_T_r = copy.deepcopy(self.robot_sim.spawn_robot_pose)  # Deep copy the robot's initial pose (position and orientation) after being spawned in the simulation
        self.w_T_robot = torch.eye(4)  # Create a 4x4 identity matrix to represent the transformation matrix of the robot in the world frame (homogeneous transformation matrix)

        # Extract the quaternion (orientation) from the robot's pose (self.w_T_r) and create a tensor for it
        # Quaternions are typically represented as (w, x, y, z)
        self.quat = torch.tensor([self.w_T_r.r.w, self.w_T_r.r.x, self.w_T_r.r.y, self.w_T_r.r.z]).unsqueeze(0)

        # Convert the quaternion to a rotation matrix (3x3 rotation matrix from the quaternion)
        self.rot = quaternion_to_matrix(self.quat)

        # Set the translation (position) part of the homogeneous transformation matrix # 
        self.w_T_robot[0, 3] = self.w_T_r.p.x  # Set x-position
        self.w_T_robot[1, 3] = self.w_T_r.p.y  # Set y-position
        self.w_T_robot[2, 3] = self.w_T_r.p.z  # Set z-position

        # Set the rotation part of the transformation matrix (top-left 3x3 block is the rotation matrix)
        self.w_T_robot[:3, :3] = self.rot[0]  # The first 3x3 block represents the rotation part

        # initiate world
        self.world_instance: World = World(self.gym, self.sim, self.env_ptr, self.world_params, w_T_r=self.w_T_r) 
            
        # get camera data:
        self.mpc_control = ReacherTask(self.task_file, self.robot_file, self.env_yml_relative_path, self.tensor_args)
        self.n_dof = self.mpc_control.controller.rollout_fn.dynamics_model.n_dofs
        self.start_qdd = torch.zeros(self.n_dof, **self.tensor_args)

        # update goal:
        self.exp_params = self.mpc_control.exp_params
        self.current_state = copy.deepcopy(self.robot_sim.get_state(self.env_ptr, self.robot_ptr))
        self.ee_list = []
        

        self.mpc_tensor_dtype = {'device':self.device, 'dtype':torch.float32}
        self.franka_bl_state = np.array([-0.3, 0.3, 0.2, -2.0, 0.0, 2.4,0.0, 0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        self.x_des_list = [self.franka_bl_state]
        self.ee_error = 10.0
        self.j = 0
        self.t_step = 0
        self.i = 0
        self.x_des = self.x_des_list[0]
        self.mpc_control.update_params(goal_state=self.x_des)

        self.x,self.y,self.z = 0.0, 0.0, 0.0
        self.asset_options = gymapi.AssetOptions()
        self.asset_options.armature = 0.001
        self.asset_options.fix_base_link = True
        self.asset_options.thickness = 0.002

        # ?
        self.object_pose = gymapi.Transform() # Represents a transform in the system. 
        self.object_pose.p = gymapi.Vec3(self.x,self.y,self.z) # Position, in meters.   
        self.object_pose.r = gymapi.Quat(0,0,0, 1) # Rotation Quaternion, represented in the format xi^ + yj^ + zk^ + w  

        self.obj_asset_file = "urdf/mug/movable_mug.urdf" # United Robot Description Format https://www.mathworks.com/help/sm/ug/urdf-model-import.html
        self.obj_asset_root = get_assets_path() # path to .../storm/content/assets
        
        # Visualizing end effector target settings 
        if(self.vis_ee_target):
            # spawn the end effector traget (goal) location (as a rigid body)
            self.target_object = self.world_instance.spawn_object(self.obj_asset_file, self.obj_asset_root, self.object_pose, color=RED, name='ee_target_object') # I assume they refer here to red cup - here they spawn it to environment of gym
            self.obj_base_handle = self.gym.get_actor_rigid_body_handle(self.env_ptr, self.target_object, 0) # ?
            self.obj_body_handle = self.gym.get_actor_rigid_body_handle(self.env_ptr, self.target_object, 6) # I assume they refer here to the "objective body" (a rigid body represents the end effector target pose (red cup))"
            self.gym.set_rigid_body_color(self.env_ptr, self.target_object, 6, gymapi.MESH_VISUAL_AND_COLLISION, RED) # giving the red cup its red color. without this row it would be gray

        # set assets path (paths?)
        self.obj_asset_file = "urdf/mug/mug.urdf"
        self.obj_asset_root = get_assets_path()

        # spawn the end effector to env
        self.ee_handle = self.world_instance.spawn_object(self.obj_asset_file, self.obj_asset_root, self.object_pose, color=RED, name='ee_current_as_mug') # end effector handle in gym env
        self.ee_body_handle = self.gym.get_actor_rigid_body_handle(self.env_ptr, self.ee_handle, 0)
        self.gym.set_rigid_body_color(self.env_ptr, self.ee_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, GREEN)
        
        # goal position and quaternion
        self.prev_mpc_goal_pos = np.ravel(self.mpc_control.controller.rollout_fn.goal_ee_pos.cpu().numpy()) # goal position
        self.prev_mpc_goal_quat = np.ravel(self.mpc_control.controller.rollout_fn.goal_ee_quat.cpu().numpy()) # goal quaternion (rotation)
        self.object_pose.p = gymapi.Vec3(self.prev_mpc_goal_pos[0], self.prev_mpc_goal_pos[1], self.prev_mpc_goal_pos[2])  # goal position
        self.object_pose.r = gymapi.Quat(self.prev_mpc_goal_quat[1], self.prev_mpc_goal_quat[2], self.prev_mpc_goal_quat[3], self.prev_mpc_goal_quat[0])  # goal quaternion (rotation)
        self.object_pose = self.w_T_r * self.object_pose
        
        if(self.vis_ee_target): 
            self.gym.set_rigid_transform(self.env_ptr, self.obj_base_handle, self.object_pose)
        self.n_dof = self.mpc_control.controller.rollout_fn.dynamics_model.n_dofs
        self.prev_acc = np.zeros(self.n_dof)
        self.ee_pose_gym_cs = gymapi.Transform()
        self.w_robot_coord = CoordinateTransform(trans=self.w_T_robot[0:3,3].unsqueeze(0),
                                            rot=self.w_T_robot[0:3,0:3].unsqueeze(0))

        self.rollout = self.mpc_control.controller.rollout_fn
        self.tensor_args = self.mpc_tensor_dtype
        self.sim_dt = self.mpc_control.exp_params['control_dt'] # frequency of control. 
    
        self.log_traj = {'q':[], 'q_des':[], 'qdd_des':[], 'qd_des':[],
                    'qddd_des':[]}

        self.q_des = None
        self.qd_des = None
        self.t_step = self.gym_instance.get_sim_time()

        self.prev_mpc_goal_pos = np.ravel(self.mpc_control.controller.rollout_fn.goal_ee_pos.cpu().numpy())
        self.prev_mpc_goal_quat = np.ravel(self.mpc_control.controller.rollout_fn.goal_ee_quat.cpu().numpy())
        
        self.name_to_handle = {} # actor names in current environment -> actor handles
        self.handle_to_name = {} # actor handles in current environment -> actor names 
        
        self.world_yml = join_path(get_gym_configs_path(), self.env_yml_relative_path) 
        self.env_template = load_yaml(self.world_yml)
        self.all_collision_objs = self.env_template['world_model']['coll_objs'] # in file
        self.non_coll_objs_actor_names = {'robot','ee_target_object', 'ee_current_as_mug'} # in file
        self.all_collision_objs_names = self.get_all_coll_obs_actor_names() # in file
    

        
    def set_name_to_handle(self, item):
        self.name_to_handle = item
        
    def set_handle_to_name(self, item):
        self.handle_to_name = item
        
    def update_goal_pose_in_mpc(self, new_goal_pose):
        """
        Informing the mpc about a change in the goal pose at the environment.
          
        new_pose: the new goal pose to pass to the mpc.
        """
        
        # register the new mpc update as the last (most recent) update
        self.prev_mpc_goal_pos[0] = new_goal_pose.p.x
        self.prev_mpc_goal_pos[1] = new_goal_pose.p.y
        self.prev_mpc_goal_pos[2] = new_goal_pose.p.z
        self.prev_mpc_goal_quat[1] = new_goal_pose.r.x
        self.prev_mpc_goal_quat[2] = new_goal_pose.r.y
        self.prev_mpc_goal_quat[3] = new_goal_pose.r.z
        self.prev_mpc_goal_quat[0] = new_goal_pose.r.w
    
        self.mpc_control.update_params(goal_ee_pos=self.prev_mpc_goal_pos,
                                    goal_ee_quat=self.prev_mpc_goal_quat)    
    
    
    def get_all_objs_with_locs(self):     
        all_coll_objs_with_locs = {}
        for obj_type in self.all_collision_objs:
            for obj_name in self.all_collision_objs[obj_type]:
                all_coll_objs_with_locs[obj_name] = self.all_collision_objs[obj_type][obj_name]
        return all_coll_objs_with_locs
    
    
        
    def step(self, at:dict, prev_at: dict):
        """
        Update arm parameters. cost_weights are the parameters for the mpc cost function.
        mpc_params are the horizon and number of particles of the mpc.
        Input
            - at: action of rlpt at time t (a(t)).
            a(t) Could either be:
                a dictionary contains the parameters to set:
                    - cost_weights: dict {cost_name: weight}
                    - mpc_params: dict {horizon: num, num_particles: num}
                a string with a special op-code telling "no action", meaning we don't
                want to perform parameter tuning (and to save time!).
            
            summary: 
                a(t) is dict <=> we perform a new hyper parameter swithing. at will containg the new values.
                a(t) is str <=> we don't perform a new hyper parameter swithing.
            
            - step_num: the time step within the episode
            
        Output
            - observation: 2 numpy arrays [object dimensions and positions], [q_pos, ee_pos, ee_quat, prev_mpc_goal_pos, g_quat]
            - reward: float reward function for RL
            - keyboard_interupt: bool - true if a keyboard interupt was detacted
        
        """
        step_start_time = time.time()  
        
        GOAL_POSE_CHANGE_TOLL = 0.0001 # TOLERANCE
        
        prev_at_exists = len(prev_at) 
        
        if update_mpc_params := (not prev_at_exists or at['mpc_params'] != prev_at['mpc_params']):
            self.mpc_control.update_mpc_params(at['mpc_params']) 
           
        if update_cost_weights := (not prev_at_exists or at['cost_weights'] !=  prev_at['cost_weights']):
            self.mpc_control.update_costs(at['cost_weights']) 
                
        self.gym_instance.step() # Advancing the simulation by one time step. TODO: I belive that should be before the cost update and not after. Check with elias

        if(self.vis_ee_target): # only when visualizng goal state (red and green cups)            
            # verified_pose_goal_gym = copy.deepcopy(self.world_instance.get_pose(self.obj_body_handle)) # exactly as in gui
            goal_handle = self.obj_body_handle
            current_goal_pose_storm_cs = self.get_body_pose(goal_handle, coordinate_system='storm') # get updated goal pose from environment (translated to  storm coordinate system)
            curr_goal_pos_storm_cs = np.ravel([current_goal_pose_storm_cs.p.x, current_goal_pose_storm_cs.p.y, current_goal_pose_storm_cs.p.z])
            curr_goal_rot_storm_cs = np.ravel([current_goal_pose_storm_cs.r.w, current_goal_pose_storm_cs.r.x, current_goal_pose_storm_cs.r.y, current_goal_pose_storm_cs.r.z])
            pos_diff_norm = np.linalg.norm(self.prev_mpc_goal_pos - curr_goal_pos_storm_cs) 
            rot_diff_norm =  np.linalg.norm(self.prev_mpc_goal_quat - curr_goal_rot_storm_cs)
            has_changed = pos_diff_norm > GOAL_POSE_CHANGE_TOLL or rot_diff_norm > GOAL_POSE_CHANGE_TOLL
            if has_changed:
                # time.sleep(200) # todo remove debug
                self.update_goal_pose_in_mpc(current_goal_pose_storm_cs) # telling mpc that goal pose has changed
        self.t_step += self.sim_dt
        
        # Get current time-step's DOF state (name, position, velocity and acceleration) for each one of the 7 dofs
        current_dofs_state_formatted = self.get_dofs_states_formatted() # updated dofs from environment. with dof names
        
        # MPC Rollouts: Plan next command* with mpc (returning not the command itself to the controller but the desired state of the dofs on the next time step)
        desired_dofs_state = self.mpc_planning(self.t_step, current_dofs_state_formatted, control_dt=self.sim_dt, WAIT=True)
        current_dofs_state_formatted_ref = current_dofs_state_formatted # not sure why we need a reference 
        current_dofs_state_tensor = torch.as_tensor(np.hstack((current_dofs_state_formatted_ref['position'], current_dofs_state_formatted_ref['velocity'], current_dofs_state_formatted_ref['acceleration'])),**self.tensor_args).unsqueeze(0)
        desired_dofs_position = copy.deepcopy(desired_dofs_state['position']) # sesired dof position for each dof (7x1 vector)
        
        # calculate costs in of state in simulator (at the "real world" not rollout costs)
        hide_collision_costs = False # to get the collision costs also in real world
        _ = self.mpc_control.get_current_error(current_dofs_state_formatted, hide_collision_costs) # ee error as storm defined it. Unused at the moment, I believe thats what running the cost function at the real world (TODO Verify it)
        # Calculate current end effector pose from the 7 current dofs states
        current_ee_pose_storm_cs = self.mpc_control.controller.rollout_fn.get_ee_pose(current_dofs_state_tensor) # see updated docs
        current_ee_pos_storm_cs = np.ravel(current_ee_pose_storm_cs['ee_pos_seq'].cpu().numpy()) # end effector position in current state
        current_ee_quat_storm_cs = np.ravel(current_ee_pose_storm_cs['ee_quat_seq'].cpu().numpy()) # end effector quaternion in current state
        
        
        # convert from storm coordinate system to gym coordinate system
        self.ee_pose_gym_cs.p = copy.deepcopy(gymapi.Vec3(current_ee_pos_storm_cs[0], current_ee_pos_storm_cs[1], current_ee_pos_storm_cs[2]))
        self.ee_pose_gym_cs.r = gymapi.Quat(current_ee_quat_storm_cs[1], current_ee_quat_storm_cs[2], current_ee_quat_storm_cs[3], current_ee_quat_storm_cs[0])
        self.ee_pose_gym_cs = copy.deepcopy(self.w_T_r) * copy.deepcopy(self.ee_pose_gym_cs) 
        
        # Update gym when using gui settings
        if(self.vis_ee_target): # Sets Transform (pose - position and quaternion) for a Rigid Body (the ee) at the environment.
            self.gym.set_rigid_transform(self.env_ptr, self.ee_body_handle, copy.deepcopy(self.ee_pose_gym_cs))
        
        if self.gui_settings['render_trajectory_lines']:
            gui_draw_lines(self.gym_instance, self.mpc_control, self.w_robot_coord) # drawing trajectory lines on screen. Can comment out
        
        # Send the command (the desired doffs position) to "controller" (to the simulator with the environment) to update the position of each dof to the desired ones (In RL terms, we could think of this like "action" here (At) is just telling the controller the next state (st+1) you want).  
        self.robot_sim.command_robot_position(desired_dofs_position, self.env_ptr, self.robot_ptr) # control dofs           
        
        
        # Save current pose in storm cs and initialize it if not initialized
        self.ee_pose_storm_cs = self.get_ee_pose_at_storm_cs_from_dofs_state(current_dofs_state_formatted_ref) # end effector position in storm cs  
        if not MpcRobotInteractive._is_initialized(MpcRobotInteractive.initial_ee_pose_storm_cs):
            MpcRobotInteractive.initial_ee_pose_storm_cs = copy.deepcopy(self.ee_pose_storm_cs)
    

        # Get the new position of the end effector in storm cs:         
        # new_dofs_state = desired_dofs_state # environment is deterministic. So next state is 100% known- (exactly the state we desired).
        return time.time() - step_start_time        
    
    @staticmethod
    def _is_initialized(var):
        return var is not None and len(var)
    
    def get_ee_pose_at_storm_cs_from_dofs_state(self, dofs_state):
        
        curr_state = np.hstack((dofs_state['position'], dofs_state['velocity'], dofs_state['acceleration']))
        curr_state_tensor = torch.as_tensor(curr_state, **self.tensor_args).unsqueeze(0) 
        ee_pose_state_storm_cs = self.mpc_control.controller.rollout_fn.get_ee_pose(curr_state_tensor) # end effector pose in storm coordinate system. see updated docs
        
        # get current pose:
        ee_pos = np.ravel(ee_pose_state_storm_cs['ee_pos_seq'].cpu().numpy())
        ee_rot = np.ravel(ee_pose_state_storm_cs['ee_quat_seq'].cpu().numpy())
        
        ee_rot_storm = copy.deepcopy(ee_rot)
        ee_rot_storm_shifted = np.array([ee_rot_storm[1], ee_rot_storm[2], ee_rot_storm[3], ee_rot_storm[0]])
        storm_pose_final = np.concatenate([ee_pos, ee_rot_storm_shifted])
     
        return storm_pose_final 
    
    def get_actor_names_indexed_at_actor_handle_from_env(self):
        """
        returns all actors in environments, indexed by handle.
        return an array "a" where for each handle i, a[i] is the actor belongs to that handle.
         
        """
        n_actors = self.gym.get_actor_count(self.env_ptr)
        actor_handles = range(n_actors)
        env_actors = [''] * len(actor_handles)
        for actor_handle in actor_handles:
            actor_name = self.gym.get_actor_name(self.env_ptr, actor_handle)
            # actor_handle_to_name[actor_handle] = actor_name
            env_actors[actor_handle] = actor_name
        return env_actors
    
    def get_actor_group_from_env(self, group_name):
        assert group_name in ['cube', 'sphere', 'robot','ee_current_as_mug',  'ee_target_object']
        all_env_actors = self.get_actor_names_indexed_at_actor_handle_from_env()
        
        ans = []
        for actor_handle, actor_name in enumerate(all_env_actors):
            if actor_name.startswith(group_name):
                
                ans.append((actor_handle, actor_name))
        return ans
    
    def get_actor_name_to_actor_handle_map(self) -> dict:
        
        n_actors = self.gym.get_actor_count(self.env_ptr)
        actor_name_to_handle = {} # reverse map
        actor_handles = range(n_actors)
        
        # map handles to actors and vise versa
        for actor_handle in actor_handles:
            actor_name = self.gym.get_actor_name(self.env_ptr, actor_handle)
            # actor_handle_to_name[actor_handle] = actor_name
            actor_name_to_handle[actor_name] = actor_handle         
        
        return actor_name_to_handle                
    
    def get_actor_handle_to_actor_name_map(self) -> dict:
        return reverse_map(self.get_actor_name_to_actor_handle_map())
        
    def reset_environment(self, selected_modified_objs_storm:dict , goal_pose_storm: List[float]=[0,0,0,0,0,0,1]):
        
        def storm_pose_to_gym_pose(storm_pose):
            return self.transform_tensor(torch.tensor(storm_pose).unsqueeze(0), self.w_T_r)

        cast_out_pose_storm = [-100] * 7 # Far away and almost invisible location. This is were we cast-out objects which we did not select by name
        cast_out_pose_gym = storm_pose_to_gym_pose(cast_out_pose_storm)    
        n_actors = self.gym.get_actor_count(self.env_ptr)
        actor_handle_to_name = [-1] * n_actors # at index i, the actor name, and i is the actor name
        actor_name_to_handle = {} # reverse map
        actor_handles = range(n_actors)
                
        # map handles to actors and vise versa
        for actor_handle in actor_handles:
            actor_name = self.gym.get_actor_name(self.env_ptr, actor_handle)
            actor_handle_to_name[actor_handle] = actor_name
            actor_name_to_handle[actor_name] = actor_handle         

        
        # select objects to include by their names from input
        env_template = self.env_template
        all_collision_objs = self.all_collision_objs
        selected_coll_objects = {'sphere': {},'cube': {}}        
        for obj_type in all_collision_objs: # sphere, cube
            for obj_name in all_collision_objs[obj_type]: # cube%x%, sphere%y%
                if obj_name in selected_modified_objs_storm.keys():
                    selected_coll_objects[obj_type][obj_name] = selected_modified_objs_storm[obj_name] 
                # if obj_name in coll_obs_names:
                #     selected_coll_objects[obj_type][obj_name] = all_collision_objs[obj_type][obj_name] 
        
        env_selected = copy.deepcopy(env_template)
        env_selected['world_model']['coll_objs'] = selected_coll_objects
        
        selected_coll_objs_poses = self.extract_poses(selected_coll_objects) 
        
        # update environment with new locations of selected objects
        self.gym.refresh_actor_root_state_tensor(self.sim) # In gym: root state (a vector in R13) is composed from: Position, Orientation, Linear Velocity, Angular Velocity
        root_state_gym_current = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim)).clone()         
        
        # object type a: collision objects to cast out to an agreed location (not participating or not selected)
        for actor_name in actor_name_to_handle:
            if actor_name not in selected_modified_objs_storm.keys() and actor_name not in ['robot', 'ee_target_object','ee_current_as_mug']:
                actor_handle = actor_name_to_handle[actor_name]
                root_state_gym_current[actor_handle, 0:7] = cast_out_pose_gym
        
        # object type b: collission objects to include (participating, selected)
        for actor_type in selected_coll_objects:
            for actor_name in selected_coll_objects[actor_type]:
                actor_handle = actor_name_to_handle[actor_name] # object idx out of all actors
                root_state_gym_current[actor_handle, 0:7] = storm_pose_to_gym_pose(selected_coll_objs_poses[actor_name]) 
        
        # object type c: goal state new location
        goal_pose_gym = storm_pose_to_gym_pose(goal_pose_storm) # storm -> gym
        root_state_gym_current[actor_name_to_handle['ee_target_object'], 0:7] = goal_pose_gym # set "root goal" to be as self.goal_pose
        self.goal_pose = goal_pose_gym.tolist()[0]
        
        # update finished. Can set the gym root tensor again to inform gym updates were made
        robot_handle = actor_name_to_handle['robot']
        range_to_robot = range(robot_handle)
        range_to_robot_tensor = torch.tensor(range_to_robot, dtype=torch.int32, device="cpu")
        range_after_robot = range(robot_handle+1, n_actors)
        range_after_robot_tensor = torch.tensor(range_after_robot,dtype=torch.int32, device="cpu")
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(root_state_gym_current), gymtorch.unwrap_tensor(range_to_robot_tensor), len(range_to_robot))
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(root_state_gym_current), gymtorch.unwrap_tensor(range_after_robot_tensor), len(range_after_robot_tensor))

        # update storm        
        self.mpc_control.update_world_params(env_selected)  
        
        # update actor name -> handle, handle->name cache
        self.set_handle_to_name(self.get_actor_handle_to_actor_name_map())
        self.set_name_to_handle(self.get_actor_name_to_actor_handle_map())
            
        return env_selected
           
    
    def episode(self, rlpt_agent:DQNAgent, episode_max_ts:int, interactive_plot=True):
        
        """
        Operating a final episode of the robot using STORM system (a final contol loop, from an initial state of the robot to some final state where its over at). 
        The control during the episode (a control loop) is done by the STORM mppi controller. 
        
        cost_weights: initial parameters for the cost function
        mpc_params: initial parameters for the cost function
        
        """
        
        if interactive_plot:
            fig = plt.figure()
            plt.ion()
             # live plotting
            at_live_ticks = 50
            at_live_plt = {'particles':[], 'pos_w': [], 'rot_w': []}
            # at2 = {'prem_col_w': [],'self_col_w': []}
            #  {'cost_weights': {'goal_pose': (1.0, 300.0), 'zero_vel': 0.0, 'zero_acc': 0.0, 'joint_l2': 0.0, 'robot_self_collision': 100, 'primitive_collision': 2000, 'voxel_collision': 0.0, 'null_space': 1.0, 'manipulability': 30, 'ee_vel': 0.0, 'stop_cost': (120.0, 1.5), 'stop_cost_acc': (0.0, 0.1), 'smooth': 1.0, 'state_bound': 1000.0}, 'mpc_params': {'horizon': 30, 'particles': 800, 'n_iters': 1}}
            
        ####################################
        ####### Setting up the episode #####
        ####################################  
        ep_idx = rlpt_agent.get_training_episodes_done()
        color_print(f"episode at index = {ep_idx}, starting now...")       
  
        # s0_as_tensor: torch.Tensor
        st: np.ndarray # 1d 
        s_next:np.ndarray # # 1d
        rt: np.float32
        at: dict
        at_id: int
        terminated:bool
        
        
        robot_handle = self.name_to_handle['robot'] # actor handle in current env
        # goal_pose_handle = self.name_to_handle['ee_target_object'] # actor handle in current env
        s0_robot_dof_states_gym = self.gym.get_actor_dof_states(self.env_ptr, robot_handle, gymapi.STATE_ALL) # TODO may need to replace by 
        s0_robot_dof_positions_gym_np = s0_robot_dof_states_gym['pos'] 
        s0_robot_dof_vels_gym_np = s0_robot_dof_states_gym['vel']
        goal_ee_pose_gym = self.get_body_pose(self.obj_body_handle, "gym") # in gym coordinate system
        goal_ee_pose_gym_np = pose_as_ndarray(goal_ee_pose_gym)
        
        # set initial mppi policy means and covariances at sniffer (empty policy)
        s0_pi_mppi_means = torch.zeros(rlpt_agent.max_horizon, 7)
        # s0_pi_mppi_means_np = torch_tensor_to_ndarray(s0_pi_mppi_means)

        # set initial mppi policy means and covariances at sniffer (empty policy)
        s0_pi_mppi_covs = torch.zeros(7)
        # s0_pi_mppi_covs_np = torch_tensor_to_ndarray(s0_pi_mppi_covs).flatten() # [1,7] to [7]

        sniffer.set_current_mppi_policy(s0_pi_mppi_means, s0_pi_mppi_covs)
        
        
        at_metadata_keyname = f'at{META_DATA_SIGNATURE_ETL}'
        st_metadata_keyname = f'st{META_DATA_SIGNATURE_ETL}'
        rt_metadata_keyname = f'rt{META_DATA_SIGNATURE_ETL}'
        snext_metadata_keyname = f's(t+1){META_DATA_SIGNATURE_ETL}'
        optim_metadata_keyname = f'optim{META_DATA_SIGNATURE_ETL}'
        step_metadata_keyname = f'step{META_DATA_SIGNATURE_ETL}' 
        logging_info = {'t_ep':[], # int
                        't_total':[], # int
                        'at_id': [], # :int
                        at_metadata_keyname: [], # dictionary {q(st,q,all): list}
                        'st':  [], # :list
                        st_metadata_keyname: [], # dictionary list'
                        'rt': [], # r(t): float
                        rt_metadata_keyname: [], # dictionary {pos:float, rot:float, safety:float, dur:float}  
                        snext_metadata_keyname: [], # dictionary: {goal: bool, self_col:bool, primitive_col:bool, ee_pos_err: float, ee_rot_err: float}                        
                        optim_metadata_keyname : [], # dictionary: {loss:float, grad:float, alpha:float, beta:float} : float
                        step_metadata_keyname: [] # dictionary: {duration:float}
                        }
       
        ####################################
        ####### Starting Episode steps #####
        ####################################
        # debug_cnt = 0
        # while gym_pose_not_initialized := math.isnan(self.get_body_pose(self.ee_body_handle, "gym").p.x):
        #     # self.gym_instance.step()            
        #     # self.gym.simulate(self.sim)
        #     # self.gym.fetch_results(self.sim, True)
        #     # self.gym.sync_frame_time(self.sim)
        #     print(f'pose not yet initialized') 
        #     print(debug_cnt)
        #     debug_cnt += 1

        prev_at = {}
        prev_at_id = -1
        st = rlpt_agent.calc_state(self, self.gym.get_actor_dof_states(self.env_ptr, robot_handle, gymapi.STATE_ALL), sniffer, prev_at_id, 0)                        
        episode_start_time = time.time()
        for ts in range(episode_max_ts):
            logging_info['t_total'].append(rlpt_agent.get_steps_done()) # total time step, all episodes

            # rlpt - select a(t) 
            at_id, at, at_metadata = rlpt_agent.select_action(as_1d_tensor(st,dtype=torch.float32))    
            print(f"debug t = {ts}, q vals: {at_metadata['q(w,st,all)']}")
            # rlpt - reset the hyper parameters and mpc planner: perform action a(t, new_parameters) in environment   
            step_duration = self.step(at, prev_at) # moving to next time step t+1, optinonally performing parameter tuning
            step_metadata = {'duration': step_duration}
            s_next = rlpt_agent.calc_state(self, self.gym.get_actor_dof_states(self.env_ptr, robot_handle, gymapi.STATE_ALL), sniffer, at_id, ts + 1)
            s_next_ee_pos = self.get_body_pose(self.ee_body_handle, "gym")
            s_next_goal_pos = self.get_body_pose(self.obj_body_handle, "gym")            
            terminated, goal_state, contact_detected, ee_pos_error, ee_rot_error = rlpt_agent.check_for_termination(sniffer, s_next_ee_pos,s_next_goal_pos,rlpt_cfg['agent']['goal_test'])
            
            # print(f'debug pos err = {ee_pos_error}, rot err = {ee_rot_error}')
            rt, rt_metadata = rlpt_agent.compute_reward(ee_pos_error, ee_rot_error, contact_detected, step_duration, goal_state, ts, episode_max_ts)    
            print(f"debug rt = {rt}")
            snext_metadata = {'pos_err':ee_pos_error, 'rot_err':ee_rot_error, 'contact':contact_detected, 'goal_state': goal_state, 'ee_pose_gym_cs':pose_as_ndarray(s_next_ee_pos)} 
            rlpt_agent.store_transition(s_next, rt, terminated) # NOTE: HERE I changed a bit compared to oiriginal code. They passed "done" (terminated or tuncated), I passed only "terminated". See https://farama.org/Gymnasium-Terminated-Truncated-Step-API#:~:text=To%20prevent%20an,for%20replicating%20work   
            
            # NoisyNet: removed decrease of epsilon
            post_step_ops_meta_data = rlpt_agent.post_step_ops(max_ts_per_episode=episode_max_ts, max_episode_index=n_episodes_real)
            optim_metadata = post_step_ops_meta_data['optimization'] 
            
            logging_info['t_ep'].append(ts)
            logging_info['at_id'].append(at_id)
            logging_info[at_metadata_keyname].append(at_metadata) 
            logging_info['st'].append(st)         
            logging_info[st_metadata_keyname].append({})
            logging_info['rt'].append(rt)
            logging_info[rt_metadata_keyname].append(rt_metadata)
            logging_info[snext_metadata_keyname].append(snext_metadata)   
            logging_info[optim_metadata_keyname].append(optim_metadata)
            logging_info[step_metadata_keyname].append(step_metadata)
            
            
        
            # escape collision if necessary (change pose until no contact)
            if not (reset_state or sample_goal_every_episode):    
                while sniffer.is_contact_real_world or sniffer.is_self_contact_real_world: # while still colliding
                    print("contact detected, escaping contact before learning continues")
                    self.step(at, prev_at)
                
    
            if not args.external_run or ((ts + 1)%(episode_max_ts/10) == 0):
                print_progress_bar((ts + 1), episode_max_ts, seconds_passed=time.time() - episode_start_time)
            
            if terminated: # terminal state reached TODO: they originaly implemented it breakout after "done" and not after "terminated" only
                color_print(f'moved from s({ts}) to s({ts+1}) but s({ts+1}) is a terminal state! (goal / collision(contact))\n\
                    goal state:{goal_state} contact detected: {contact_detected}')
                color_print(f'termination reward: {rt}')
                break
            
            if interactive_plot:    
                # at_live_plt['particles'].append(at['mpc_params']['particles'])
                # at_live_plt['rot_w'].append(at['cost_weights']['goal_pose'][0])
                # at_live_plt['pos_w'].append(at['cost_weights']['goal_pose'][1])
                
                if ts % at_live_ticks == 0:
                    fig.clear()
                    plt.plot(logging_info['at_id'],label='action id')
                    plt.plot(logging_info['rt'],label='reward')
                    # plt.plot(at_live_plt['particles'],label='particles')
                    # plt.plot(at_live_plt['rot_w'],label='rot_w')
                    # plt.plot(at_live_plt['pos_w'],label='pos_w')
                    plt.legend()
                    plt.title(f'episode {ep_idx} run time: {time.time()-episode_start_time}')
                    plt.pause(0.000000000000001)
                    
            
            st = s_next
            prev_at = at
            prev_at_id = at_id
            
            
            
        
        ########## end of episode ###########
        losses = [optim_metadata['loss'] for optim_metadata in logging_info[optim_metadata_keyname]]
        all_q_vals_all_transitions = [atmd["q(w,st,all)"] for atmd in logging_info[at_metadata_keyname]] 
        q_vals_means = [np.mean(qvals) for qvals in all_q_vals_all_transitions]
        q_vals_mean = np.mean(q_vals_means)
        
        episode_dur_sec = time.time() - episode_start_time      
        color_print(f'episode at index {ep_idx} finished!')
        color_print(f'- completed steps: {(ts + 1)}, seconds in total- {episode_dur_sec}, average step duration: {episode_dur_sec/ (ts + 1)}')
        color_print(f'- rewards: total- {sum(logging_info["rt"])}, average- {np.mean(logging_info["rt"])}')
        color_print(f'- losses: total- {sum(losses)}, average: {np.mean(losses)}')
        color_print(f'- current buffer size: {len(rlpt_agent.memory)}')
        color_print(f'- mean q val in all visited states: {q_vals_mean}')
        
        
        # print(f'debug 1: {len(logging_info["t_ep"])}, {len(logging_info["at_id"])}')
        return logging_info
        
          
            
        
    def get_all_coll_obs_actor_names(self): # all including non participating
        all_names = self.get_actor_name_to_actor_handle_map().keys()
        non_col_objs = self.non_coll_objs_actor_names
        ans = set()
        for name in all_names:
            if name not in non_col_objs:
                if name in ans:
                    raise BadArgumentUsage("user has to pass different collision obsjet names")  
                ans.add(name)
        return list(ans)
                
    

    
            
    # helper functions:    
    def get_body_pose(self, handle, coordinate_system) -> gymapi.Transform:
        """
            Getting most updated pose of a body by its handle (for example if you override its pose in gui, that pose is updated immediately at next sim rendering step)
        """
        gym_pose = copy.deepcopy(self.world_instance.get_pose(handle)) # gym coordinates first. Exactly as seen in gui (verified)
        if coordinate_system == 'storm':
            storm_pose = copy.deepcopy(self.w_T_r.inverse() * gym_pose) 
            return storm_pose # translate to storm coordinates
        elif coordinate_system == 'gym':
            return gym_pose
        else:
            raise BadArgumentUsage("should pass storm or gym")                 
    def mpc_planning(self,t_step, current_dofs_states_formatted, control_dt, WAIT):
        """
        A WRAPPER: Made this to improve code explainability without changing original function name.        
        """
        # Generate rollouts and return the desired state of each dof of the robot (7 dofs in total) 

        return self.mpc_control.get_command(t_step, current_dofs_states_formatted, control_dt, WAIT) # next command to send to controller
    def get_dofs_states_formatted(self):
        """
        A WRAPPER: Made this to improve code explainability without changing original function name
        """
        return copy.deepcopy(self.robot_sim.get_state(self.env_ptr, self.robot_ptr)) # calling to the original function 
    def extract_poses(self, dictionary):
        # poses = []
        poses = {}
        for obj_type, obj_data in dictionary.items():
            if isinstance(obj_data, dict):
                for obj_name, obj_info in obj_data.items():
                    if 'pose' in obj_info:
                        # poses.append(obj_info['pose'])
                        poses[obj_name] = obj_info['pose'] 
                    if 'position' in obj_info:
                        poses[obj_name] = obj_info['position'] + [0.0, 0.0, 0.0, 1]
                        # poses.append(obj_info['position'] + [0.0, 0.0, 0.0, 1])
        return poses
    def transform_tensor(self, tensor, w_T_r):
        """
        Transorming a pose (vector in length 7 (3 position, 4 rotation)) from STORM coordinate system to GUI coordinate system.
        
        Parameters:

        tensor: A PyTorch tensor containing rows of pose data. Each row consists of a position (x, y, z) and a quaternion (x, y, z, w).
        w_T_r: A transformation (gymapi.Transform) representing the relationship between two coordinate frames (e.g., world frame to robot frame).
        Workflow:

        Loop Over Rows: Iterates over each row in the input tensor, where each row represents a pose (position + quaternion).
        Create Pose: Converts each row into a gymapi.Transform object using the Vec3 (position) and Quat (quaternion) classes.
        Apply Transformation: Multiplies w_T_r with the pose. This applies the transformation, effectively converting the pose into a new coordinate frame (e.g., transforming an object's pose from the robot frame to the world frame).
        Store Transformed Pose: Extracts the position and quaternion from the transformed pose and appends it to the transformed_tensor list.
        Return Result: Converts the list of transformed poses into a PyTorch tensor and returns it.
        Purpose: Transforms a list of poses from one coordinate frame to another using the transformation w_T_r.
        This function is used when you need to apply a transformation to update the poses based on the current simulation state.
        
        might help: https://mecharithm.com/learning/lesson/homogenous-transformation-matrices-configurations-in-robotics-12
        https://www.quora.com/What-is-a-homogeneous-transformation-matrix 
        
        
        """

        transformed_tensor = []

        for row in tensor:
            pose = gymapi.Transform()
            pose.p = gymapi.Vec3(row[0], row[1], row[2])
            pose.r = gymapi.Quat(row[3], row[4], row[5], row[6])
            
            table_pose = w_T_r * pose # pose in STORM coordinates -> pose in GUI coordinates 

            transformed_row = [table_pose.p.x, table_pose.p.y, table_pose.p.z,
                            table_pose.r.x, table_pose.r.y, table_pose.r.z, table_pose.r.w]
            
            transformed_tensor.append(transformed_row)

        return torch.tensor(transformed_tensor)
    def convert_to_transform(self, pose):
        pose_t = gymapi.Transform()
        pose_t.p = gymapi.Vec3(pose[0], pose[1], pose[2])
        pose_t.r = gymapi.Quat(pose[3], pose[4], pose[5], pose[6])
        return pose_t
    def generate_random_position(self, n):
        # Generate random position in a grid divided into n^2 blocks
        x = random.uniform(-2/n, 2/n)
        y = random.uniform(-2/n, 2/n)
        z = random.uniform(0.1, 0.7)
        return [x, y, z]
    def generate_excluded_random_position(self):
        # Randomly decide which range to use
        x = 0
        y = 0
        if random.random() < 0.5:
            # Generate a random number in the range -1 to -0.2
            x =  random.uniform(-1, -0.15)
        else:
            # Generate a random number in the range 0.2 to 1
            x = random.uniform(0.15, 1)
        if random.random() < 0.5:
            # Generate a random number in the range -1 to -0.2
            y =  random.uniform(-1, -0.15)
        else:
            # Generate a random number in the range 0.2 to 1
            y = random.uniform(0.15, 1)
        z = random.uniform(0.1, 0.7)
        return [x, y, z]
    def generate_random_quaternion(self):
        # Generate random quaternion representing a rotation
        euler_angles = [random.uniform(0, 2*math.pi) for _ in range(3)]
        rotation = R.from_euler('xyz', euler_angles)
        quaternion = rotation.as_quat()
        return quaternion.tolist()
    def open_yaml(self, yml_file):
        with open(yml_file) as file:
            return yaml.load(file, Loader=yaml.FullLoader)
    def get_objects_by_indexes(self, world_params, indexes):
        coll_objs = world_params['world_model']['coll_objs'] # spheres and cubes   
        # Flatten the dictionary into a list of (key, value) pairs
        objects = []
        for obj_type, obj_dict in coll_objs.items():
            for obj_name, obj_info in obj_dict.items():
                objects.append((obj_name, obj_info))
        
        # Get the objects corresponding to the provided indexes
        selected_objects = []
        if indexes is None:
            selected_objects = objects
        else:
            for index in indexes:
                if 0 <= index < len(objects):
                    selected_objects.append(objects[index])
                else:
                    raise IndexError(f"Index {index} out of range")
            
        return selected_objects
    def get_base_name(self, name):
        base_name = ''.join([char for char in name if char.isalpha()])
        return base_name
    def randomize_pos(self, obj, base_name):

        position = self.generate_excluded_random_position()
        if base_name == 'cube':
            quat = self.generate_random_quaternion()
            return position + quat
        else:
            return position                
    
def add_padding_to_objs(participating, not_participating):
    all_objs = participating + not_participating
def select_obj_poses(participating, all_coll_objs_with_positions) -> Tuple[dict, dict]:
    
    """

    generates a new world for each episode.must return a tuple of:
    
    """
    
    def modify_coll_obj_inplace_rec(collision_obj):
        """
        add noise to location of collision objects

        Args:
            collision_obj (_type_): _description_
        """
        eps = 1e-4
        for i, item in enumerate(collision_obj): 
            if isinstance(item, Iterable):
                modify_coll_obj_inplace_rec(item)
            else:
                sd = np.mean(np.abs(collision_obj))
                if sd == 0:
                    sd = eps
                noise = np.random.normal(0,sd,1)[0]
                collision_obj[i] = collision_obj + noise
    
    def is_sphere(obj):
        return 'radius' in obj
    def is_cube(obj):
        return 'dims' in obj
    
    
    
    very_large = 100 # in meters            
    far_away_position = [very_large] * 3
    default_orient = [0] * 4
    far_away_pose = far_away_position + default_orient
    very_small = 1e-4 # in meters
    invisible_cube_dims = [very_small] * 3
    external_to_env_state_sphere = {'radius': very_small, 'position': far_away_position}
    external_to_env_state_cube = {'dims': invisible_cube_dims , 'pose': far_away_pose}
    modify_coll_obj_inplace_rec(participating)
    not_participatig = {} # name to loc
    for obj_name in all_coll_objs_with_positions:
        if obj_name not in participating:
            curr_obj = all_coll_objs_with_positions[obj_name]
            not_participatig[obj_name] = external_to_env_state_cube if is_cube(curr_obj) else external_to_env_state_sphere 
    return participating, not_participatig                           
def sample_obj_subset(all_coll_objs_with_positions):
    
    # all collision objects which are available 
    all_obj_names = all_coll_objs_with_positions.keys()
    
    # select a subset of objects from the available to participate in the environment (or stay with all)
    total_objs = len(all_coll_objs_with_positions)
    participating_number = random.randint(1, total_objs)
    # particiating = sorted(random.sample(all_obj_names, participating_number))
    participating_names = random.sample(all_obj_names, participating_number)
    participating = {obj_name: all_coll_objs_with_positions[obj_name] for obj_name in participating_names} # name to loc 

    return participating

def episode_loop(n_episodes, episode_max_ts, cfg,training=True):
    """
    Running episodes in a loop.
    Args:
        n_episodes (_type_): number of 
        episode_max_sim_ts (_type_): _description_
        
    Returns:
        _type_: _description_
    """
    
    # load configuration
    
    def query_episodes_done(rlpt_agent:Union[rlptAgentBase, None]):
        if rlpt_agent is None:
            return 0
        elif rlpt_agent.training_mode:
            return rlpt_agent.get_training_episodes_done()
        else:
            return rlpt_agent.get_test_episodes_done()
        
    mpc = mpc_ri      
    sample_objs_every_episode = cfg['sample_objs_every_episode'] 
    sample_obj_locs_every_episode = cfg['sample_obj_locs_every_episode']
    goal_pose_storm = cfg['default_goal_pose'] # in storm coordinates
    all_col_objs_with_locs = mpc.get_all_objs_with_locs() # all collision objects (participating or not) with their locations
    rlpt_agent = None
    
    while query_episodes_done(rlpt_agent) < n_episodes:
        # re-define participating and non-participating objects 
        particiating_storm = all_col_objs_with_locs
        not_participatig_storm = {}
        if sample_objs_every_episode:
            particiating_storm = sample_obj_subset(all_col_objs_with_locs)
        if sample_obj_locs_every_episode:
            particiating_storm, not_participatig_storm = select_obj_poses(particiating_storm, all_col_objs_with_locs) 
        # re-define ee goal pose
        if sample_goal_every_episode:        
            goal_pose_storm = [-0.37, -0.37, 0.3, 0, 2.5, 0, 1] if query_episodes_done(rlpt_agent) % 2 == 0 else list(MpcRobotInteractive.initial_ee_pose_storm_cs / 2)  # in storm coordinates
        # reset environment with selections
        _ = mpc.reset_environment(particiating_storm, goal_pose_storm) # reset environment and return its new specifications
       
       
        ###### pre-processing (only at the beginning) #####
        if query_episodes_done(rlpt_agent) == 0:
            # Initialize the rlpt agent
            all_col_objs_handles_list = mpc.get_actor_group_from_env('cube') + mpc.get_actor_group_from_env('sphere') # [(name i , name i's handle)]  
            rainbow_agent_params = {**rlpt_cfg['agent']['rainbow_agent_settings']}
            rainbow_agent_params['super_params'] =  {
                'base_pos_gym': np.array(list(mpc.gym.get_actor_rigid_body_states(mpc.env_ptr,mpc.name_to_handle['robot'],gymapi.STATE_ALL)[0][0][0])), # [0][0] is [base link index][pose index][pos]
                'participating_storm': particiating_storm,
                'not_participating_storm': not_participatig_storm,
                'col_obj_handles':  {pair[1]:pair[0] for pair in all_col_objs_handles_list}, # {obj name (str): obj handle (int)} 
                'action_space':rlpt_action_space, 
                'training_mode':training,
                'etl_logging': include_etl,
                'reward_cfg': rlpt_cfg['agent']['reward']
            } 
            
            rlpt_agent = DQNAgent(**rainbow_agent_params) # rainbow agent
            
            # load model from checkpoint
            if load_checkpoint_model:
                _ = rlpt_agent.load(model_file_path) # loading and inializing the checkpoint model 

        ###### run next episode ######
        assert rlpt_agent is not None # only to remove red error underlines (no worries, it wont be done at this point)
        episode_logging_info = mpc.episode(rlpt_agent, episode_max_ts,interactive_plot=rlpt_cfg['gui']['interactive_plot'])  # type: ignore
        rlpt_agent.post_episode_ops(model_file_path,etl_file_path=etl_file_path,episode_logging_info=episode_logging_info)

    
if __name__ == '__main__':
    
    ############################
    ###### Pre Processing ######
    ############################
    
    # parse arguments to start simulation
    parser = argparse.ArgumentParser(description='pass args')
    parser.add_argument('--robot', type=str, default='franka', help='Robot to spawn') # robot name
    parser.add_argument('--cuda', action='store_true', default=True, help='use cuda') # use cude
    parser.add_argument('--headless', action='store_true', default=False, help='headless gym') # False means use viewer (gui)
    parser.add_argument('--control_space', type=str, default='acc', help='Robot to spawn')
    # parser.add_argument('--env_yml_relative', type=str, default='', help='asset specifications of environment. Relative path under storm/content/configs/gym')
    # parser.add_argument('--physics_engine_yml_relative', type=str, default='', help='physics specifications of environment. Relative path under storm/content/configs/gym')
    # parser.add_argument('--task_yml_relative', type=str, default='', help='task specifications. Relative path under storm/content/configs/mpc')
    parser.add_argument('--rlpt_cfg_path', type=str,default='BGU/Rlpt/configs/main.yml', help= 'config file of rl parameter tuner')
    # external run params:
    parser.add_argument('--external_run', type=bool, default=False, help= 'run from external script')
    parser.add_argument('--model_path', type=str, default='', help= 'path to model file')    
    args = parser.parse_args()
    
    # simulation setup
    
                        
    # rlpt setup
    if not GLobalVars.is_defined('rlpt_cfg'):
        GLobalVars.rlpt_cfg = load_config_with_defaults(args.rlpt_cfg_path)
        rlpt_cfg = GLobalVars.rlpt_cfg
    task_file = rlpt_cfg['external_cfgs']['task_yml_relative']
    physics_engine_config = load_yaml(join_path(get_gym_configs_path(),rlpt_cfg['external_cfgs']['physics_engine_yml_relative']))
    sim_params = physics_engine_config.copy() # GYM DOCS/Simulation Setup — Isaac Gym documentation.pdf
    sim_params['headless'] = rlpt_cfg['gui']['headless'] # run with no gym gui
    include_etl = rlpt_cfg['agent']['model']['include_etl']
    env_file = rlpt_cfg['external_cfgs']['env_yml_relative']
                
    sniffer_params:dict = copy.deepcopy(rlpt_cfg['cost_sniffer'])
    GLobalVars.cost_sniffer = CostFnSniffer(**sniffer_params)
    sniffer = GLobalVars.cost_sniffer 

    profile_memory = rlpt_cfg['profile_memory']['include'] # activate memory profiling
    if profile_memory: # for debugging gpu if needed   
        start_mem_profiling()   
    
    # define rlpt action space
    rlpt_action_space = make_rlpt_actionspace()
    
    
    gym = Gym(**sim_params) # note - only one initiation is allowed per process!!! (p.n: I tried latelty for a very long time to re-create this instance, but failed to release the gpu memory it used)
    mpc_ri = MpcRobotInteractive(args, gym, rlpt_cfg, env_file, task_file) # not the best naming.             
    
    # make/get model (nn) and etl paths for the rlpt 
    if args.external_run:
        model_file_path = args.model_path
        load_checkpoint_model = os.path.exists(model_file_path) # if the model file exists, load it. Else, we'll make one
    else:
        load_checkpoint_model = rlpt_cfg['agent']['model']['load_checkpoint']
        if load_checkpoint_model:  
            model_file_path = rlpt_cfg['agent']['model']['checkpoint_path']
            assert os.path.exists(model_file_path)        
        else:
            assert rlpt_cfg['agent']['training']['run'], 'testing cannot innitiate a model'
            model_file_path = make_model_path(rlpt_cfg['agent']['model']['dst_dir'])
            
    model_dir = os.path.split(model_file_path)[0]

    
    #############################
    ######### Training ##########
    #############################
    
    ep_loop_cfg = rlpt_cfg['agent']['training']
    if ep_loop_cfg['run']:
        n_episodes_real = ep_loop_cfg['n_episodes']
        sample_goal_every_episode = ep_loop_cfg['sample_goal_every_episode']
        one_episode_max_ts = rlpt_cfg['agent']['training']['max_ts']
        all_episodes_max_ts = n_episodes_real * one_episode_max_ts  
        
        if include_etl:
            etl_file_path = f'{model_dir}/{ep_loop_cfg["etl_file_name"]}'
        reset_state = ep_loop_cfg['reset_to_initial_state_every_episode']
        if args.external_run:
            save_checkpoints_every_episode = True
            if reset_state:
                n_episodes_loop = 1
            else:
                n_episodes_loop = n_episodes_real
        else: # internal run 
            save_checkpoints_every_episode = rlpt_cfg['agent']['training']['save_checkpoints']
            n_episodes_loop = n_episodes_real 
            assert not reset_state, 'must run from external'
             
        if (save_checkpoints_every_episode or include_etl) and not os.path.exists(model_dir):
            os.mkdir(model_dir)
            

        episode_loop(n_episodes_loop, one_episode_max_ts, ep_loop_cfg)

    
    #################################################################
    ######### Testing (greedy policy and no optimization) ###########
    #################################################################
    ep_loop_cfg = rlpt_cfg['agent']['testing']
    if ep_loop_cfg['run']:
        n_episodes_real = ep_loop_cfg['n_episodes']  
        sample_goal_every_episode = ep_loop_cfg['sample_goal_every_episode']
        save_checkpoints_every_episode = False
        if include_etl:
            etl_file_path = f'{model_dir}/{ep_loop_cfg["etl_file_name"]}'
        episodes_cfg = rlpt_cfg['agent']['testing']
        reset_state = ep_loop_cfg['reset_to_initial_state_every_episode']
        if args.external_run:
            assert os.path.exists(model_file_path), 'model path must exist'
            if reset_state:
                n_episodes_loop = 1 # to perform a beginning from the same initial state every time, we'll run many episode-loops of one episode at a time in each loop.      
        
            else:
                n_episodes_loop = n_episodes_real
        else:
            n_episodes_loop = n_episodes_real 
            assert not reset_state, 'must run from external'
            
        episode_loop(n_episodes_loop, rlpt_cfg['agent']['testing']['max_ts'], episodes_cfg, training=False)
        
    
    if profile_memory:
        finish_mem_profiling(rlpt_cfg['profile_memory']['pickle_path'])
    
    exit(0)
