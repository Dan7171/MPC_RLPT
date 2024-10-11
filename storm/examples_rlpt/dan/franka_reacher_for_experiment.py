 
""" 
Based on Elias's franka_reacher_for_comparison.py 
"""

import copy
from os import name
import pickle
from re import I, match
from select import select
import shutil
from typing import Iterable, List, Tuple, Union
from click import BadArgumentUsage
from colorlog import root
from cv2 import norm
from isaacgym import gymapi
from isaacgym import gymutil
from isaacgym import gymtorch
from matplotlib.transforms import Transform
from storm_kit.mpc.cost import cost_base
from sympy import Integer, im
import torch
from traitlets import default
from BGU.Rlpt.reward import point_cloud_utils
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
import BGU.Rlpt.reward.point_cloud_utils 
import matplotlib.pyplot as plt
from BGU.Rlpt.experiments.experiment_utils import get_combinations
from BGU.Rlpt.experiments.experiment_utils import make_date_time_str
np.set_printoptions(precision=2)
GREEN = gymapi.Vec3(0.0, 0.8, 0.0)
RED = gymapi.Vec3(0.8, 0.1, 0.1)
 
 
def get_actor_name(gym, env, actor_handle):
    return gym.get_actor_name(env, actor_handle)
def make_plot(x:Union[None,tuple]=None, ys:list=[]):
    # figure: The top level container for all the plot elements.
    # Axes: An Axes object encapsulates all the elements of an individual (sub-)plot in a figure.
    # pyplot: matplotlib.pyplot is a state-based interface to matplotlib. It provides an implicit, MATLAB-like, way of plotting. It also opens figures on your screen, and acts as the figure GUI manager.
    y_labels = [''] * len(ys)
    
    if x is not None and x[1] is not None: # x label passed
        plt.xlabel(x[1])
        
    for i,y in enumerate(ys):
        y_values = y[0]
        y_label = y[1]
        
        if x is None or x[0] is None: # did not pass x values  
            plt.plot(y_values)
        else: # passed x values
            plt.plot(x[1], y_values)
    
        y_labels[i] = y_label 
    plt.legend(y_labels, loc="upper right")        
    plt.show()
    
# Functions for converting gym objects to numpy vectors
def pose_as_ndarray(pose:gymapi.Transform) -> np.ndarray:
        """Converting a pose from a Transform object to a np.array in length 7 (indices 0-2 = position, 3-6 = rotation) """
        # get pos and rot as ndarray
        pos_np = pos_as_ndarray(pose.p)
        rot_np = rot_as_ndarray(pose.r)
        
        # concatenate to one vector in length 7
        return np.concatenate([pos_np, rot_np]) 
def pos_as_ndarray(pos:gymapi.Vec3) -> np.ndarray:
    
    """
    cast pos from gymapi.Vec3 to an ndarray in length 3 (np array - vector)
    """
    
    return np.array([pos.x, pos.y, pos.z])   
def rot_as_ndarray(rot:gymapi.Quat) -> np.ndarray:
    
    """
    cast rot from gymapi.Quat to an ndarray in length 4 (np array - vector)
    """
    
    return np.array([rot.x, rot.y, rot.z, rot.w])

# Error measurment functions:
def pose_error(curr_pose:gymapi.Transform, goal_pose:gymapi.Transform)-> np.float64:
    """
    return l2 norm between current and desired poses (each pose is both positon and rotation)  
    """
    return np.linalg.norm(pose_as_ndarray(curr_pose) - pose_as_ndarray(goal_pose))  
def pos_error(curr_pos:gymapi.Vec3, goal_pos:gymapi.Vec3)-> np.float64:
    """
    return l2 norm between current and desired positions (position is only the spacial location in environment ("x,y,z" coordinates))
    """
    return np.linalg.norm(pos_as_ndarray(curr_pos) - pos_as_ndarray(goal_pos))     
def rot_error(curr_rot:gymapi.Quat, goal_rot:gymapi.Quat)-> np.float64:
    """
    return l2 norm between current and desired rotations (each rotation is the quaternion - a 4 length vector)  
    """
    
    return np.linalg.norm(rot_as_ndarray(curr_rot) - rot_as_ndarray(goal_rot)) 
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
    torch.cuda.memory._record_memory_history(max_entries=100000)
def finish_mem_profiling(output_path):
    """
    stop profiling memory and save it to output path
    drag and drop output file here to analyze: https://pytorch.org/memory_viz
    """
    
    torch.cuda.memory._dump_snapshot(output_path)    
    print(f"memory usage profile was saved to {output_path}")
class MpcRobotInteractive:
    """
    This class is for controlling the arm base and simulator.
    It contains the functions for RL learning.
    Operations to control the simulation
    TODO: Don't really like this class. This class should be called "Controller" or "Actions for simulation" since thats what it is
    """
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

        # # define obstacles in world
        # self.table_dims = np.ravel([1.5,2.5,0.7])
        # self.cube_pose = np.ravel([0.35, -0.0,-0.35,0.0, 0.0, 0.0,1.0])
        # self.cube_pose = np.ravel([0.9,0.3,0.4, 0.0, 0.0, 0.0,1.0])
        # self.table_dims = np.ravel([0.35,0.1,0.8])    
        # self.cube_pose = np.ravel([0.35,0.3,0.4, 0.0, 0.0, 0.0,1.0])
        # self.table_dims = np.ravel([0.3,0.1,0.8])
    

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
        self.object_pose = gymapi.Transform() # Represents a transform in the system. https://drive.google.com/file/d/17bDctjlUkYOetNksOd4Ux3k34duq7Zqs/view?usp=sharing
        self.object_pose.p = gymapi.Vec3(self.x,self.y,self.z) # Position, in meters. https://drive.google.com/file/d/17bDctjlUkYOetNksOd4Ux3k34duq7Zqs/view?usp=sharing
        self.object_pose.r = gymapi.Quat(0,0,0, 1) # Rotation Quaternion, represented in the format xi^ + yj^ + zk^ + w  https://drive.google.com/file/d/17bDctjlUkYOetNksOd4Ux3k34duq7Zqs/view?usp=sharing

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
        self.sim_dt = self.mpc_control.exp_params['control_dt']
    
        self.log_traj = {'q':[], 'q_des':[], 'qdd_des':[], 'qd_des':[],
                    'qddd_des':[]}

        self.q_des = None
        self.qd_des = None
        self.t_step = self.gym_instance.get_sim_time()

        self.prev_mpc_goal_pos = np.ravel(self.mpc_control.controller.rollout_fn.goal_ee_pos.cpu().numpy())
        self.prev_mpc_goal_quat = np.ravel(self.mpc_control.controller.rollout_fn.goal_ee_quat.cpu().numpy())
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
    def step(self, cost_weights, mpc_params, step_num: int):
        """
        Update arm parameters. cost_weights are the parameters for the mpc cost function.
        mpc_params are the horizon and number of particles of the mpc.
        Input
            - cost_weights: dict {cost_name: weight}
            - mpc_params: dict {horizon: num, num_particles: num}
            - step_num: the time step within the episode
        Output
            - observation: 2 numpy arrays [object dimensions and positions], [q_pos, ee_pos, ee_quat, prev_mpc_goal_pos, g_quat]
            - reward: float reward function for RL
            - keyboard_interupt: bool - true if a keyboard interupt was detacted
        
        """
        GOAL_POSE_CHANGE_TOLL = 0.0001 # TOLERANCE
                    
        # Update Cost and MPC variables dynamically (RLPT Part)
        self.mpc_control.update_costs(cost_weights) 
        if step_num == 0: # technical issue - loading to gpu takes time. So we do it only at the beginning for now.
            self.mpc_control.update_mpc_params(mpc_params) 
        
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
        
        # Don't miss (even if its commented out):

        # # Get the new position of the end effector in storm's cs:         
        # new_dofs_state = desired_dofs_state # environment is deterministic. So next state is 100% known- (exactly the state we desired).
        # self.ee_pose_storm_cs = self.get_ee_pose_at_storm_cs_from_dofs_state(new_dofs_state) # end effector position in storm cs  
    def get_ee_pose_at_storm_cs_from_dofs_state(self, dofs_state):
        
        curr_state = np.hstack((dofs_state['position'], dofs_state['velocity'], dofs_state['acceleration']))
        curr_state_tensor = torch.as_tensor(curr_state, **self.tensor_args).unsqueeze(0) 
        ee_pose_state_storm_cs = self.mpc_control.controller.rollout_fn.get_ee_pose(curr_state_tensor) # end effector pose in storm coordinate system. see updated docs
        
        # get current pose:
        e_pos = np.ravel(ee_pose_state_storm_cs['ee_pos_seq'].cpu().numpy())
        e_quat = np.ravel(ee_pose_state_storm_cs['ee_quat_seq'].cpu().numpy())
        
        return e_pos, e_quat 
    
    def reset_environment(self, coll_obs_names: List[str]=[] , goal_pose_storm: List[float]=[0,0,0,0,0,0,1]):
        
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
        if coll_obs_names == []: # if input list is empty, take all objets in file 
            coll_obs_names = actor_handle_to_name                

        # robot_idx = 0
        # ee_target_object_idx = -2
        # ee_current_as_mug_idx = -1
        
        # select objects to include by their names from input
        world_yml = join_path(get_gym_configs_path(), self.env_yml_relative_path) 
        env_template = load_yaml(world_yml)
        all_collision_objs = env_template['world_model']['coll_objs']
        selected_coll_objects = {'sphere': {},'cube': {}}        
        for obj_type in all_collision_objs: # sphere, cube
            for obj_name in all_collision_objs[obj_type]: # cube%x%, sphere%y%
                if obj_name in coll_obs_names:
                    selected_coll_objects[obj_type][obj_name] = all_collision_objs[obj_type][obj_name] 
        env_selected = copy.deepcopy(env_template)
        env_selected['world_model']['coll_objs'] = selected_coll_objects
        selected_coll_objs_poses = self.extract_poses(selected_coll_objects) 
        
        # update environment with new locations of selected objects
        self.gym.refresh_actor_root_state_tensor(self.sim) # In gym: root state (a vector in R13) is composed from: Position, Orientation, Linear Velocity, Angular Velocity
        root_state_gym_current = gymtorch.wrap_tensor(self.gym.acquire_actor_root_state_tensor(self.sim)).clone()         
        
        # object type a: collision objects to cast out to an agreed location
        for actor_name in actor_name_to_handle:
            if actor_name not in coll_obs_names and actor_name not in ['robot', 'ee_target_object','ee_current_as_mug']:
                actor_handle = actor_name_to_handle[actor_name]
                root_state_gym_current[actor_handle, 0:7] = cast_out_pose_gym
        
        # object type b: collission objects to include
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
        return env_selected
           
    def goal_test(self, pos_error:np.float64, rot_error:np.float64, eps=1e-3):
        return pos_error < eps and rot_error < eps     
    def episode(self, cost_weights, mpc_params,episode_max_ts, ep_num=0) -> Tuple[int, float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        
        """
        Operating a final episode of the robot using STORM system (a final contol loop, from an initial state of the robot to some final state where its over at). 
        The control during the episode (a control loop) is done by the STORM mppi controller. 
        
        cost_weights: initial parameters for the cost function
        mpc_params: initial parameters for the cost function
        tuning: TODO: serving the rlpt. if True, the cost and mpc params will be re-selected throughout the episode frequently (probably on every time step). 
        
        Return:
        a tuple t
            t[0](int) is the number of time-steps took took to reach goal position & orientation. -1 if failed to reach  
            t[1](float) is the total run time of episde in seconds
            
        """
        cost_weights = copy.deepcopy(cost_weights) # to ensure a call by value and not by reference (we don't want to modify the params outside of the function)
        
        # document all errors along the episode 
        pos_errors = np.zeros(episode_max_ts)
        rot_errors = np.zeros(episode_max_ts)
        self_collision_errors = np.zeros(episode_max_ts)
        objs_collision_errors = np.zeros(episode_max_ts)
        
        
        # -- start episode control loop --
        ep_start_time = time.time()
        for ts in range(episode_max_ts):
            
            # Planning current step with rollouts and executing it in environment 
            print(f"episode: {ep_num} time step: {ts} ")
            # mpc.step(cost_weights, mpc_params, ts) 
            self.step(cost_weights, mpc_params, ts)
            # calculate orientation and position errors and perform convergence to goal state:
            curr_ee_pose: gymapi.Transform = self.get_body_pose(self.ee_body_handle, "gym") # in gym coordinate system
            goal_ee_pose: gymapi.Transform = self.get_body_pose(self.obj_body_handle, "gym") # in gym coordinate system
            ee_pos_error: np.float64 = pos_error(curr_ee_pose.p, goal_ee_pose.p) # end effector position error
            ee_rot_error: np.float64 = rot_error(curr_ee_pose.r,goal_ee_pose.r)  # end effector rotation error   
            
            # taking storm's undweighted collision cost terms as a measure for the how approximate the robot is to obstacles or to collide with itslef  
            # these cost terms are using robot-links point clouds and environment objects point clouds to punish robot for being too close to the links 
            # note: unlike whats written in paper, the result is not 0 or 1. But its   
            cost_terms_current_step:dict = GLobalVars.cost_sniffer.get_current_costs() # current real world costs
            unweighted_cost_self_coll:np.float32 = np.ravel(cost_terms_current_step['self_collision'].term.cpu().numpy())[0] # robot with itself collision cost (unweighted)
            unweighted_cost_primitive_coll: np.float32 = np.ravel(cost_terms_current_step['primitive_collision'].term.cpu().numpy())[0] # robot with objects in environment collision cost (unweighted)             
            # robot_pt = point_cloud_utils.generate_robot_point_cloud(mpc.env_ptr, mpc.gym)
    
            # save errors of the whole episode 
            pos_errors[ts] = ee_pos_error 
            rot_errors[ts] = ee_rot_error
            self_collision_errors[ts] = unweighted_cost_self_coll
            objs_collision_errors[ts] = unweighted_cost_primitive_coll
            
            
            # goal test
            reached_goal = self.goal_test(ee_pos_error, ee_rot_error) 
            if reached_goal: 
                break
            # storm_costs:dict = GLobalVars.cost_sniffer.get_current_costs()
            # for name, vals in storm_costs.items():
            #     print(name, vals)
        # -- end of episode --
       
        if reached_goal:
            time_to_goal = time.time() - ep_start_time
            return ts, time_to_goal, pos_errors, rot_errors, self_collision_errors, objs_collision_errors
        else:
            return -1, -1,  pos_errors, rot_errors, self_collision_errors, objs_collision_errors    
    # helper functions:    
    def get_body_pose(self, handle, coordinate_system) -> gymapi.Transform:
        """
            Getting most updated pose of a body by its handler (for example if you override its pose in gui, that pose is updated immediately at next sim rendering step)
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
    def select_random_indexes(self, first, last, max_n, min_n):
        """
        first and last indexes
        n = max number of objects
        min = min number of objects
        """
        #print("ELIAS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        # Define the range of indexes from first to last (inclusive)
        indexes = list(range(first, last))
        
        # Randomly determine how many indexes to select (between 0 and n)
        num_indexes_to_select = random.randint(min_n, max_n)
        
        # Randomly select the determined number of indexes from the list
        selected_indexes = random.sample(indexes, min(num_indexes_to_select, len(indexes)))
        
        return selected_indexes
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
    def select_participating_obstacles(self, world_yml, indexes_cubes=[], indexes_spheres=[], render_poses=True):
        """
        # Formerly called "modify_dict" but changed to this name for more clarity.
         
        This method: 
        1. selects randomly just a subset of the participating obstacles in next run, out of all available obstacles (cubes and spheres) in  world_yml
        2. changing randomly the selected obstacles locations, using randomize_pos()

        """
        
        MIN_SPHERE_INDEX = 0 # inclusive
        MAX_SPHERE_INDEX = 9 # inclusive
        # MIN_CUBE_INDEX = 11 # inclusive
        MIN_CUBE_INDEX = 10 # inclusive
        MAX_CUBE_INDEX = 34 # inclusive 
        # MIN_CUBE_INDEX = 0
         #MAX_CUBE_INDEX = 25
        is_sampling_cubes, is_sampling_spheres = indexes_cubes == [], indexes_spheres == []
         
        if not is_sampling_cubes: # selelecting cube indices manually
            assert all([MIN_CUBE_INDEX <= ind <= MAX_CUBE_INDEX for ind in indexes_cubes])
        else: # sample cubes
            min_cubes = 1 # # minimum cubes to select
            max_cubes = 20 # maximum cubes to select
            indexes_cubes = self.select_random_indexes(MIN_CUBE_INDEX, MAX_CUBE_INDEX + 1, max_cubes, min_cubes)
        
        # same for the spheres
        if not is_sampling_spheres: # selelecting cube indices manually
            assert all([MIN_SPHERE_INDEX <= ind <= MAX_SPHERE_INDEX for ind in indexes_spheres])
        else:
            min_spheres = 1 # minumum spheres to select
            max_spheres = 10 # maximum spheres to select
            indexes_spheres = self.select_random_indexes(MIN_SPHERE_INDEX, MAX_SPHERE_INDEX + 1, max_spheres, min_spheres)  
        
        
        indexes = indexes_spheres + indexes_cubes
        world_params = self.open_yaml(world_yml) # = {'world_model': {'coll_objs': {'sphere': {all spheres..},'cube': {all cubes..}}}}
        selected_objects = self.get_objects_by_indexes(world_params, indexes)
        compressed_world_params = {'world_model': {'coll_objs': {'sphere': {},'cube': {}}}}
        sphere_index = 1
        cube_index = 1
        
        for i in range(len(indexes)):
            obj = selected_objects[i]
            name = obj[0]
            base_name = self.get_base_name(name)
            
            if render_poses:
                new_pos = self.randomize_pos(obj, base_name) 
            
            if base_name == 'sphere':
                # Modify dict
                if render_poses:
                    world_params['world_model']['coll_objs'][base_name][name]['position'] = new_pos
                # Add to compressed dict
                radius_position = {}
                radius_position['radius'] = world_params['world_model']['coll_objs'][base_name][name]['radius']
                radius_position['position'] = world_params['world_model']['coll_objs'][base_name][name]['position']
                compressed_world_params['world_model']['coll_objs'][base_name][base_name + str(sphere_index)] = radius_position
                sphere_index += 1
            elif base_name == 'cube':
                #print("Cube added !!!")
                # Modify dict
                if render_poses:
                    world_params['world_model']['coll_objs'][base_name][name]['pose'] = new_pos
                # Add to compressed dict
                dims_pose = {}
                dims_pose['dims'] = world_params['world_model']['coll_objs'][base_name][name]['dims']
                dims_pose['pose'] = world_params['world_model']['coll_objs'][base_name][name]['pose']
                compressed_world_params['world_model']['coll_objs'][base_name][base_name + str(cube_index)] = dims_pose
                cube_index += 1

            dims_pose = {}
            dims_pose['dims'] = world_params['world_model']['coll_objs']['cube']['cube28']['dims']
            dims_pose['pose'] = world_params['world_model']['coll_objs']['cube']['cube28']['pose']
            compressed_world_params['world_model']['coll_objs']['cube']['cube28'] = dims_pose

        
        return world_params, indexes, compressed_world_params # return: all optional obstacles objects
    

def main_experiment():
    
    
    # EPISODES = 1 # How many simulations / episodes to run in total
    # episode_max_ts = 800 # maximal number of time steps in a single episode 
    # FIGURE_COLUMNS = 1 
    
    # 3(env) x 3(self col) x 3(prim col) x 3 (smooth) x 3 (H) x 3 (particles) x 3 (n_iter) x  3 x 3 300
    # 
    episode_settings_space = {    
        'episode_max_ts': [30000],
        'goal_pose_storm' :[[0.47, 0.47, 0.1, 0, 2.5, 0, 1]], # [[0.47, 0.47, 0.1, 0, 0, 0, 1]]'goal_pose_storm': [[0, 0, 0.51, 0, 0, 0, 1],[0.4,-0.5, 0.3, 0, 0, 0, 1],[-0.2,-0.5, 0.3, 0, 0, 0, 1]],
        'coll_obs_names': [[]] # 'coll_obs_names': [[], ['sphere1','sphere2']] # [] means all
        
        }
    
    cost_weights_space = { 
                
        # "goal_pose": [[15.0, 100.0], [100, 1000], [500,1000], [1000,1000]], # orientation, position
        "goal_pose":  [[15.0, 100.0]],
        "zero_vel": [0.0], 
        "zero_acc": [0.0],
        "joint_l2": [0.0],
        
        # ArmBase costs
        "robot_self_collision": [5000],
        # "robot_self_collision" : [5000, 1000, 3000, 10000],  # 5000, 
        "primitive_collision": [1],
        # "primitive_collision" :  [5000, 1000, 3000, 10000],# 5000,
        "voxel_collision" : [0],
        "null_space": [1.0],
        "manipulability": [30],
        # "manipulability": [30, 10, 50, 100], 
        "ee_vel": [0.0], 
        # "stop_cost": [100, 10, 50, 200], 
        "stop_cost": [100],
        "stop_cost_acc": [0.0], 
        "smooth": [1.0],
        # "smooth": [1.0, 0.1, 5, 10], 
        # "state_bound": [1000.0, 100, 500, 2000]
        "state_bound": [1000.0],
        # "state_bound": [200.0] # Joint limit avoidance
    }
    mpc_params_space = {
        "horizon" : [30] , #  How deep into the future each rollout (imaginary simulation) sees
        # "particles" : [500, 50, 100, 1000, 2000],# How many rollouts are done. from paper:Number of trajectories sampled per iteration of optimization (or particles)
        "particles": [500],
        # "n_iters": [1, 3, 5] # Num of optimization steps - TODO (from paper)
        "n_iters": [1]
        }
    
    all_combos = get_combinations({
        'cost_weights': get_combinations(cost_weights_space),
        'mpc_params': get_combinations(mpc_params_space),
        'episode_settings': get_combinations(episode_settings_space)
        })
    
    
    
    # parse arguments to start simulation
    parser = argparse.ArgumentParser(description='pass args')
    parser.add_argument('--robot', type=str, default='franka', help='Robot to spawn') # robot name
    parser.add_argument('--cuda', action='store_true', default=True, help='use cuda') # use cude
    parser.add_argument('--headless', action='store_true', default=False, help='headless gym') # False means use viewer (gui)
    parser.add_argument('--control_space', type=str, default='acc', help='Robot to spawn')
    parser.add_argument('--env_yml_relative', type=str, default='', help='asset specifications of environment. Relative path under storm/content/configs/gym')
    parser.add_argument('--physics_engine_yml_relative', type=str, default='', help='physics specifications of environment. Relative path under storm/content/configs/gym')
    parser.add_argument('--task_yml_relative', type=str, default='', help='task specifications. Relative path under storm/content/configs/mpc')
    parser.add_argument('--rlpt_cfg_path', type=str,default='BGU/Rlpt/configs/main.yml', help= 'config file of rl parameter tuner')
    args = parser.parse_args()
    # simulation setup
    if args.physics_engine_yml_relative == '':
        args.physics_engine_yml_relative = 'rlpt/experiments/experiment1/physx.yml'    
    if args.task_yml_relative == '':
        args.task_yml_relative = 'rlpt/experiments/experiment1/franka_reacher.yml'    
    if args.env_yml_relative == '':
        args.env_yml_relative = 'rlpt/experiments/experiment1/spheres_only_general.yml'
    physics_engine_config = load_yaml(join_path(get_gym_configs_path(),args.physics_engine_yml_relative))
    sim_params = physics_engine_config.copy() # GYM DOCS/Simulation Setup — Isaac Gym documentation.pdf
    sim_params['headless'] = args.headless # run with no gym gui
                    
    # rlpt setup
    rlpt_cfg = load_config_with_defaults(args.rlpt_cfg_path)
    sniffer_params:dict = copy.deepcopy(rlpt_cfg['cost_sniffer'])
    GLobalVars.cost_sniffer = CostFnSniffer(**sniffer_params)
    profile_memory = rlpt_cfg['profile_memory']['include'] # activate memory profiling
        
    ##### main loop of episodes execution: #######
    
    gym = Gym(**sim_params) # note - only one initiation is allowed per process
    env_file = args.env_yml_relative
    task_file = args.task_yml_relative
    output_file = f'BGU/Rlpt/experiments/experiments_results/experiment1_{make_date_time_str()}.pl'
    
    try:
        
        if profile_memory: # for debugging gpu if needed   
            start_mem_profiling()   
            
        for ep, combo in enumerate(all_combos): # for each episode id with a unique combination of initial parameters
            # define episode settings
            episode_max_ts: int = combo['episode_settings']['episode_max_ts']
            if ep == 0:
                mpc = MpcRobotInteractive(args, gym, rlpt_cfg, env_file, task_file) # not the best naming. TODO:  # run episode
                data = []
                
            goal_pose_storm: list = combo['episode_settings']['goal_pose_storm']
            coll_obs_names: list = combo['episode_settings']['coll_obs_names']
            episode_environment_model = mpc.reset_environment(coll_obs_names, goal_pose_storm) # reset environment and return its new specifications
            selected_collision_objs = episode_environment_model['world_model']['coll_objs'] 
            mpc_params: dict = combo['mpc_params']
            cost_weights: dict = combo['cost_weights']
            
            # run episode and collect data  
            steps_to_goal, time_to_goal, pos_errors, rot_errors, self_col_errors, obj_col_errors = mpc.episode(cost_weights,mpc_params, episode_max_ts,ep_num=ep) 
            
            # save collected data from episode
            if ep > 0:
                with open(output_file, 'rb') as file:
                    data = pickle.load(file)    
            episode_data = {
                'settings': {'goal_pose_storm': goal_pose_storm, ' collision_objs': selected_collision_objs, 'mpc_params': mpc_params, 'cost_weights': cost_weights},
                'results': {'steps_to_goal': steps_to_goal , 'time_to_goal': time_to_goal, 'pos_errors':pos_errors, 'rot_errors':rot_errors , 'self_col_errors':self_col_errors, 'obj_col_errors':obj_col_errors}
            }
            data.append(episode_data) # index specifies the episode id
            with open(output_file, 'wb') as file:
                pickle.dump(data, file)
                
        if profile_memory:
            finish_mem_profiling(rlpt_cfg['profile_memory']['pickle_path'])

        
    except torch.cuda.OutOfMemoryError: 
        if profile_memory:
            finish_mem_profiling(rlpt_cfg['profile_memory']['pickle_path'])
  
    
    
if __name__ == '__main__':
    
    main_experiment()
    