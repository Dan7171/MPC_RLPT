#
# MIT License
#
# Copyright (c) 2020-2021 NVIDIA CORPORATION.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.#
""" Example spawning a robot in gym 

"""
import copy
from isaacgym import gymapi
from isaacgym import gymutil

from isaacgym import gymtorch
from isaacgym.gymapi import Tensor
print("Imported isaacgym gymtorch !!!!!!!!!!!!!!!!!!!!!!!!!!!")

import torch
torch.multiprocessing.set_start_method('spawn',force=True)
torch.set_num_threads(8)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
#



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

np.set_printoptions(precision=2)

world_params_list = [{'world_model': {'coll_objs': {'sphere': {'sphere1': {'radius': 0.1, 'position': [0.4, 0.4, 0.1]}}, 'cube': {'cube1': {'dims': [0.3, 0.1, 0.4], 'pose': [0.4, 0.2, 0.2, 0, 0, 0, 1.0]}, 'cube2': {'dims': [0.3, 0.1, 0.5], 'pose': [0.4, -0.3, 0.2, 0, 0, 0, 1.0]}, 'cube3': {'dims': [0.3, 0.1, 0.5], 'pose': [-1.85, 0.4, -2.95, 0, 0, 1.0, 0.0]}, 'cube4': {'dims': [0.1, 0.1, 0.1], 'pose': [-0.55, 0.57, 0.06, 0, 0, 1.0, 0.0]}, 'cube5': {'dims': [2.0, 2.0, 0.2], 'pose': [0.0, 0.0, -0.1, 0, 0, 0, 1.0]}}}}}
, {'world_model': {'coll_objs': {'sphere': {'sphere1': {'radius': 0.1, 'position': [0.4, 0.4, 0.1]}}, 'cube': {'cube1': {'dims': [0.3, 0.1, 0.4], 'pose': [0.4, 0.2, 0.2, 0, 0, 0, 1.0]}, 'cube2': {'dims': [0.3, 0.1, 0.5], 'pose': [0.4, -0.3, 0.7, 0, 0, 0, 1.0]}, 'cube3': {'dims': [0.3, 0.1, 0.5], 'pose': [-1.85, 0.4, -1.95, 0, 0, 1.0, 0.0]}, 'cube4': {'dims': [0.1, 0.1, 0.1], 'pose': [-0.55, 0.57, 1.06, 0, 0, 1.0, 0.0]}, 'cube5': {'dims': [2.0, 2.0, 0.2], 'pose': [0.0, 0.0, -0.1, 0, 0, 0, 1.0]}}}}}]

def reset(objects, arm_pos, goal_pos):
    """
    Set objects in environment, arm in initial position and target goal
    Input
        - objects: dict {object_type: [pos, dimension]}
        - arm_pos: numpy array [n_joint], initial joint configuration
        - goal_pos: numpy array [7], (x,y,z, quaternion) of target
    Output
        - observation: 2 numpy arrays [object dimensions and positions], [q_pos, ee_pos, ee_quat]
    """
    # return observation
    pass

def get_reward():
    pass
def set_goal(ee_pos, ee_quat):
    """ Set goal position.
    """
    pass
def set_params(cost_params, mpc_params):
    """ Set cost and mpc parameters.
        [ee_pos, ee_quat]
    """
    pass
def get_invalid_config():
    pass
def get_goal_reached():
    pass


class MpcRobotInteractive:
    """
    This class is for controlling the arm base and simulator.
    It contains the functions for RL learning.
    """
    def __init__(self, args, gym_instance):
        self.args = args
        self.gym_instance = gym_instance
        
        # RL variables
        self.end_effector_pos = None
        self.end_effector_quat = None
        self.goal_pose = [0,0,0,0,0,0,1]
        self.arm_configuration = None
        self.objects_configuration = None

        # File variables
        self.vis_ee_target = True
        self.robot_file = self.args.robot + '.yml'
        self.task_file = self.args.robot + '_reacher.yml'
        self.world_file = 'collision_primitives_3d_origina2.yml'

        # Simulator variables
        self.pose = None # Goal pose in simulation

        self.gym = self.gym_instance.gym
        self.sim = self.gym_instance.sim
        self.world_yml = join_path(get_gym_configs_path(), self.world_file)

        with open(self.world_yml) as file:
            self.world_params = yaml.load(file, Loader=yaml.FullLoader)
        print(f"self.world_params: {self.world_params}")

        self.robot_yml = join_path(get_gym_configs_path(),self.args.robot + '.yml')
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
        self.env_ptr = self.gym_instance.env_list[0]
        self.robot_ptr = self.robot_sim.spawn_robot(self.env_ptr, self.robot_pose, coll_id=2)

        self.device = torch.device('cuda', 0) 

    
        self.tensor_args = {'device':self.device, 'dtype':torch.float32}
    

        # spawn camera:
        self.robot_camera_pose = np.array([1.6,-1.5, 1.8,0.707,0.0,0.0,0.707])
        self.q = as_float_array(from_euler_angles(-0.5 * 90.0 * 0.01745, 50.0 * 0.01745, 90 * 0.01745))
        self.robot_camera_pose[3:] = np.array([self.q[1], self.q[2], self.q[3], self.q[0]])

    
        self.robot_sim.spawn_camera(self.env_ptr, 60, 640, 480, self.robot_camera_pose)

        # get pose
        self.w_T_r = copy.deepcopy(self.robot_sim.spawn_robot_pose)
        
        self.w_T_robot = torch.eye(4)
        self.quat = torch.tensor([self.w_T_r.r.w,self.w_T_r.r.x,self.w_T_r.r.y,self.w_T_r.r.z]).unsqueeze(0)
        self.rot = quaternion_to_matrix(self.quat)
        self.w_T_robot[0,3] = self.w_T_r.p.x
        self.w_T_robot[1,3] = self.w_T_r.p.y
        self.w_T_robot[2,3] = self.w_T_r.p.z
        self.w_T_robot[:3,:3] = self.rot[0]

        self.world_instance = World(self.gym, self.sim, self.env_ptr, self.world_params, w_T_r=self.w_T_r)
    

    
        self.table_dims = np.ravel([1.5,2.5,0.7])
        self.cube_pose = np.ravel([0.35, -0.0,-0.35,0.0, 0.0, 0.0,1.0])
        


        self.cube_pose = np.ravel([0.9,0.3,0.4, 0.0, 0.0, 0.0,1.0])
        
        self.table_dims = np.ravel([0.35,0.1,0.8])

    
    
        self.cube_pose = np.ravel([0.35,0.3,0.4, 0.0, 0.0, 0.0,1.0])
        
        self.table_dims = np.ravel([0.3,0.1,0.8])
    

        # get camera data:
        self.mpc_control = ReacherTask(self.task_file, self.robot_file, self.world_file, self.tensor_args)

        self.n_dof = self.mpc_control.controller.rollout_fn.dynamics_model.n_dofs

        
        self.start_qdd = torch.zeros(self.n_dof, **self.tensor_args)

        # update goal:

        self.exp_params = self.mpc_control.exp_params
        
        self.current_state = copy.deepcopy(self.robot_sim.get_state(self.env_ptr, self.robot_ptr))
        self.ee_list = []
        

        self.mpc_tensor_dtype = {'device':self.device, 'dtype':torch.float32}

        self.franka_bl_state = np.array([-0.3, 0.3, 0.2, -2.0, 0.0, 2.4,0.0,
                                    0.0,0.0,0.0,0.0,0.0,0.0,0.0])
        self.x_des_list = [self.franka_bl_state]
        
        self.ee_error = 10.0
        self.j = 0
        self.t_step = 0
        self.i = 0
        self.x_des = self.x_des_list[0]
        
        self.mpc_control.update_params(goal_state=self.x_des)

        # spawn object:
        self.x,self.y,self.z = 0.0, 0.0, 0.0
        self.tray_color = gymapi.Vec3(0.8, 0.1, 0.1)
        self.asset_options = gymapi.AssetOptions()
        self.asset_options.armature = 0.001
        self.asset_options.fix_base_link = True
        self.asset_options.thickness = 0.002


        self.object_pose = gymapi.Transform()
        self.object_pose.p = gymapi.Vec3(self.x,self.y,self.z)
        self.object_pose.r = gymapi.Quat(0,0,0, 1)
    
        self.obj_asset_file = "urdf/mug/movable_mug.urdf" 
        self.obj_asset_root = get_assets_path()
    
        if(self.vis_ee_target):
            self.target_object = self.world_instance.spawn_object(self.obj_asset_file, self.obj_asset_root, self.object_pose, color=self.tray_color, name='ee_target_object')
            self.obj_base_handle = self.gym.get_actor_rigid_body_handle(self.env_ptr, self.target_object, 0)
            self.obj_body_handle = self.gym.get_actor_rigid_body_handle(self.env_ptr, self.target_object, 6)
            self.gym.set_rigid_body_color(self.env_ptr, self.target_object, 0, gymapi.MESH_VISUAL_AND_COLLISION, self.tray_color)
            self.gym.set_rigid_body_color(self.env_ptr, self.target_object, 6, gymapi.MESH_VISUAL_AND_COLLISION, self.tray_color)


        self.obj_asset_file = "urdf/mug/mug.urdf"
        self.obj_asset_root = get_assets_path()


        self.ee_handle = self.world_instance.spawn_object(self.obj_asset_file, self.obj_asset_root, self.object_pose, color=self.tray_color, name='ee_current_as_mug')
        self.ee_body_handle = self.gym.get_actor_rigid_body_handle(self.env_ptr, self.ee_handle, 0)
        self.tray_color = gymapi.Vec3(0.0, 0.8, 0.0)
        self.gym.set_rigid_body_color(self.env_ptr, self.ee_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, self.tray_color)


        self.g_pos = np.ravel(self.mpc_control.controller.rollout_fn.goal_ee_pos.cpu().numpy())
        
        self.g_q = np.ravel(self.mpc_control.controller.rollout_fn.goal_ee_quat.cpu().numpy())
        self.object_pose.p = gymapi.Vec3(self.g_pos[0], self.g_pos[1], self.g_pos[2])

        self.object_pose.r = gymapi.Quat(self.g_q[1], self.g_q[2], self.g_q[3], self.g_q[0])
        self.object_pose = self.w_T_r * self.object_pose
        if(self.vis_ee_target):
            self.gym.set_rigid_transform(self.env_ptr, self.obj_base_handle, self.object_pose)
        self.n_dof = self.mpc_control.controller.rollout_fn.dynamics_model.n_dofs
        self.prev_acc = np.zeros(self.n_dof)
        self.ee_pose = gymapi.Transform()
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

        self.g_pos = np.ravel(self.mpc_control.controller.rollout_fn.goal_ee_pos.cpu().numpy())
        self.g_q = np.ravel(self.mpc_control.controller.rollout_fn.goal_ee_quat.cpu().numpy())

    def step(self, cost_params, mpc_params, i):
        """
        Update arm parameters. cost_params are the parameters for the mpc cost function. mpc_params are the horizon and number of particles of the mpc.
        Input
            - cost_params: dict {cost_name: weight}
            - mpc_params: dict {horizon: num, num_particles: num}
        Output
            - observation: 2 numpy arrays [object dimensions and positions], [q_pos, ee_pos, ee_quat, g_pos, g_quat]
            - reward: float reward function for RL
            - done: bool True, False. True if the arm reached te goal or if it is in an invalid configuration
        """

        try:
            ####################################################
            # Update Cost and MPC variables dynamically ########
            #print(f"Updating cost_params: {cost_params}")
            self.mpc_control.update_costs(cost_params)
            if i%500 == 0:
                #print(f"Updating MPC parameters: {mpc_params}")
                self.mpc_control.update_mpc_params(mpc_params)
            ####################################################
            ####################################################

            self.gym_instance.step()
            if(self.vis_ee_target):
                if self.pose == None:
                    self.pose = copy.deepcopy(self.world_instance.get_pose(self.obj_body_handle))
                    self.pose = copy.deepcopy(self.w_T_r.inverse() * self.pose)
                self.update_pose()
                #self.goal_pose = copy.deepcopy(pose) # RL

                if(np.linalg.norm(self.g_pos - np.ravel([self.pose.p.x, self.pose.p.y, self.pose.p.z])) > 0.00001 or (np.linalg.norm(self.g_q - np.ravel([self.pose.r.w, self.pose.r.x, self.pose.r.y, self.pose.r.z]))>0.0)):
                    self.g_pos[0] = self.pose.p.x
                    self.g_pos[1] = self.pose.p.y
                    self.g_pos[2] = self.pose.p.z
                    self.g_q[1] = self.pose.r.x
                    self.g_q[2] = self.pose.r.y
                    self.g_q[3] = self.pose.r.z
                    self.g_q[0] = self.pose.r.w
                    print(f"self.g_pos[0: {self.g_pos[0]}")

                    self.mpc_control.update_params(goal_ee_pos=self.g_pos,
                                              goal_ee_quat=self.g_q)
            self.t_step += self.sim_dt
            
            current_robot_state = copy.deepcopy(self.robot_sim.get_state(self.env_ptr, self.robot_ptr))

            
            command = self.mpc_control.get_command(self.t_step, current_robot_state, control_dt=self.sim_dt, WAIT=True)

            filtered_state_mpc = current_robot_state #mpc_control.current_state
            curr_state = np.hstack((filtered_state_mpc['position'], filtered_state_mpc['velocity'], filtered_state_mpc['acceleration']))

            curr_state_tensor = torch.as_tensor(curr_state, **self.tensor_args).unsqueeze(0)
            # get position command:
            q_des = copy.deepcopy(command['position'])
            qd_des = copy.deepcopy(command['velocity']) #* 0.5
            qdd_des = copy.deepcopy(command['acceleration'])
            
            ee_error = self.mpc_control.get_current_error(filtered_state_mpc)
             
            pose_state = self.mpc_control.controller.rollout_fn.get_ee_pose(curr_state_tensor)
            
            # get current pose:
            e_pos = np.ravel(pose_state['ee_pos_seq'].cpu().numpy())
            e_quat = np.ravel(pose_state['ee_quat_seq'].cpu().numpy())

            self.ee_pose.p = copy.deepcopy(gymapi.Vec3(e_pos[0], e_pos[1], e_pos[2]))
            self.ee_pose.r = gymapi.Quat(e_quat[1], e_quat[2], e_quat[3], e_quat[0])
            
            self.ee_pose = copy.deepcopy(self.w_T_r) * copy.deepcopy(self.ee_pose)
            
            if(self.vis_ee_target):
                self.gym.set_rigid_transform(self.env_ptr, self.ee_body_handle, copy.deepcopy(self.ee_pose))

            print(["{:.3f}".format(x) for x in ee_error], "{:.3f}".format(self.mpc_control.opt_dt),
                  "{:.3f}".format(self.mpc_control.mpc_dt))
        
            
            self.gym_instance.clear_lines()
            top_trajs = self.mpc_control.top_trajs.cpu().float()#.numpy()
            n_p, n_t = top_trajs.shape[0], top_trajs.shape[1]
            w_pts = self.w_robot_coord.transform_point(top_trajs.view(n_p * n_t, 3)).view(n_p, n_t, 3)


            top_trajs = w_pts.cpu().numpy()
            color = np.array([0.0, 1.0, 0.0])
            for k in range(top_trajs.shape[0]):
                pts = top_trajs[k,:,:]
                color[0] = float(k) / float(top_trajs.shape[0])
                color[1] = 1.0 - float(k) / float(top_trajs.shape[0])
                self.gym_instance.draw_lines(pts, color=color)
            
            self.robot_sim.command_robot_position(q_des, self.env_ptr, self.robot_ptr)
            #robot_sim.set_robot_state(q_des, qd_des, env_ptr, robot_ptr)
            current_state = command

            self.arm_configuration = copy.deepcopy(current_state["position"])
            self.update_end_effector_pose(current_state)


        except KeyboardInterrupt:
            print('Closing')
            done = True
            self.mpc_control.close()
            return None,None,None,None,True

        return self.arm_configuration, self.goal_pose, self.end_effector_pos, self.end_effector_quat, False


    def update_end_effector_pose(self, current_state):
        curr_state = np.hstack((current_state['position'], current_state['velocity'], current_state['acceleration']))

        curr_state_tensor = torch.as_tensor(curr_state, **self.tensor_args).unsqueeze(0)
            
        pose_state = self.mpc_control.controller.rollout_fn.get_ee_pose(curr_state_tensor)
        
        # get current pose:
        e_pos = np.ravel(pose_state['ee_pos_seq'].cpu().numpy())
        e_quat = np.ravel(pose_state['ee_quat_seq'].cpu().numpy())

        self.end_effector_pos = copy.deepcopy(e_pos) # RL
        self.end_effector_quat = copy.deepcopy(e_quat) # RL
    
    def update_pose(self):
        self.pose.p.x = self.goal_pose[0]
        self.pose.p.y = self.goal_pose[1]
        self.pose.p.z = self.goal_pose[2]
        self.pose.r.x = self.goal_pose[3]
        self.pose.r.y = self.goal_pose[4]
        self.pose.r.z = self.goal_pose[5]
        self.pose.r.w = self.goal_pose[6]

        self.pose = copy.deepcopy(self.w_T_r.inverse() * self.pose)
        #self.pose = copy.deepcopy(self.w_T_r * self.pose)


    def reset(self):
        """
        Change location of objects in environment and target goal
        Input
            - objects: dict {object_type: [pos, dimension]}
            - goal_pos: numpy array [7], (x,y,z, quaternion) of target
        Output
            - observation: 2 numpy arrays [object dimensions and positions], [q_pos, ee_pos, ee_quat]
        """
        print("Resetting !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        world_yml = join_path(get_gym_configs_path(), self.world_file)
        world_params, indexes, compressed_world_params = modify_dict(world_yml)
        # refresh observation
        self.gym.refresh_actor_root_state_tensor(self.sim)

        # acquire root state tensor descriptor
        _root_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)

        # wrap it in a PyTorch Tensor
        root_tensor = gymtorch.wrap_tensor(_root_tensor)

        # save a copy of the original root states
        saved_root_tensor = root_tensor.clone()
        root_positions = saved_root_tensor[1:-2, 0:7]
        #print(f"root_positions: {root_positions.shape}")
       
        # Extract new object poses
        poses = extract_poses(world_params['world_model']['coll_objs'])

        # Create a torch tensor from the poses
        root_positions[:, 0:7] = transform_tensor(torch.tensor(poses), self.w_T_r) 
        #saved_root_tensor[2:-2, 0:7] *= 0

        # Set new goal
        p = generate_random_position(3)
        q = generate_random_quaternion()
        #self.goal_pose = p + q
        print(f"transform_tensor(torch.tensor(p + q).unsqueeze(0), self.w_T_r).tolist(): {transform_tensor(torch.tensor(p + q).unsqueeze(0), self.w_T_r).tolist()}")
        self.goal_pose = transform_tensor(torch.tensor(p + q).unsqueeze(0), self.w_T_r).tolist()[0]
        self.update_pose()
        print(f"self.goal_pose: {self.goal_pose}")

        root_goal = saved_root_tensor[39, 0:7]
        print(f"self.w_T_r): {self.w_T_r}!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        root_goal[0:7] = torch.tensor(self.goal_pose)

        #root_positions[0, 0:7] = torch.tensor(self.goal_pose)
        #root_goal[0:7] = transform_inverse_tensor(torch.tensor(self.goal_pose).unsqueeze(0), self.w_T_r) #self.goal_pose
        #root_goal[0:7] = torch.tensor([self.pose.p.x, self.pose.p.y, self.pose.p.z,
                           #self.pose.r.x, self.pose.r.y, self.pose.r.z, self.pose.r.w])
        #print("Tensor size:", root_positions.size())
        #print("Tensor:")
        #print(saved_root_tensor)

        # Update simulation object positions
        num_points = 37
        int_linspace = np.linspace(1, 37, num=num_points, dtype=int)
        actor_indices = torch.tensor(np.append(int_linspace, 39), dtype=torch.int32, device="cpu")
        print(f"saved_root_tensor: {saved_root_tensor.shape} !!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(f"actor_indices: {actor_indices.shape} !!!!!!!!!!!!!!!!!!!!!!!!!!")
        self.gym.set_actor_root_state_tensor_indexed(self.sim, gymtorch.unwrap_tensor(saved_root_tensor), gymtorch.unwrap_tensor(actor_indices), 38)

        # Update world_params
        self.mpc_control.update_world_params(compressed_world_params)


###############################################################################################
#####################h HELPER FUNCTIONS #######################################################
def extract_poses(dictionary):
    poses = []
    for obj_type, obj_data in dictionary.items():
        if isinstance(obj_data, dict):
            for obj_name, obj_info in obj_data.items():
                if 'pose' in obj_info:
                    poses.append(obj_info['pose'])
                if 'position' in obj_info:
                    poses.append(obj_info['position'] + [0.0, 0.0, 0.0, 1])
    return poses

def transform_tensor(tensor, w_T_r):
    transformed_tensor = []

    for row in tensor:
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(row[0], row[1], row[2])
        pose.r = gymapi.Quat(row[3], row[4], row[5], row[6])
        
        table_pose = w_T_r * pose

        transformed_row = [table_pose.p.x, table_pose.p.y, table_pose.p.z,
                           table_pose.r.x, table_pose.r.y, table_pose.r.z, table_pose.r.w]
        
        transformed_tensor.append(transformed_row)

    return torch.tensor(transformed_tensor)

def transform_inverse_tensor(tensor, w_T_r):
    transformed_tensor = []

    for row in tensor:
        pose = gymapi.Transform()
        pose.p = gymapi.Vec3(row[0], row[1], row[2])
        pose.r = gymapi.Quat(row[3], row[4], row[5], row[6])
        
        table_pose =   w_T_r.inverse() * pose

        transformed_row = [table_pose.p.x, table_pose.p.y, table_pose.p.z,
                           table_pose.r.x, table_pose.r.y, table_pose.r.z, table_pose.r.w]
        
        transformed_tensor.append(transformed_row)

    return torch.tensor(transformed_tensor)

def convert_to_transform(pose):
    pose_t = gymapi.Transform()
    pose_t.p = gymapi.Vec3(pose[0], pose[1], pose[2])
    pose_t.r = gymapi.Quat(pose[3], pose[4], pose[5], pose[6])
    return pose_t

def select_random_indexes(first, last, n):
    #print("ELIAS !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    # Define the range of indexes from first to last (inclusive)
    indexes = list(range(first, last))
    
    # Randomly determine how many indexes to select (between 0 and n)
    num_indexes_to_select = random.randint(1, n)
    
    # Randomly select the determined number of indexes from the list
    selected_indexes = random.sample(indexes, min(num_indexes_to_select, len(indexes)))
    
    return selected_indexes

def generate_random_position(n):
    # Generate random position in a grid divided into n^2 blocks
    x = random.uniform(-2/n, 2/n)
    y = random.uniform(-2/n, 2/n)
    z = random.uniform(0.1, 0.7)
    return [x, y, z]

def generate_random_quaternion():
    # Generate random quaternion representing a rotation
    euler_angles = [random.uniform(0, 2*math.pi) for _ in range(3)]
    rotation = R.from_euler('xyz', euler_angles)
    quaternion = rotation.as_quat()
    return quaternion.tolist()

def open_yaml(world_yml):
    with open(world_yml) as file:
        world_params = yaml.load(file, Loader=yaml.FullLoader)
    #print(f"world_params: {world_params}")
    return world_params

def get_objects_by_indexes(world_params, indexes):
    coll_objs = world_params['world_model']['coll_objs']
    
    # Flatten the dictionary into a list of (key, value) pairs
    objects = []
    for obj_type, obj_dict in coll_objs.items():
        for obj_name, obj_info in obj_dict.items():
            objects.append((obj_name, obj_info))
    
    # Get the objects corresponding to the provided indexes
    selected_objects = []
    for index in indexes:
        if 0 <= index < len(objects):
            selected_objects.append(objects[index])
        else:
            raise IndexError(f"Index {index} out of range")
    
    return selected_objects

def get_base_name(name):
    base_name = ''.join([char for char in name if char.isalpha()])
    return base_name

def randomize_pos(obj, base_name):

    position = generate_random_position(2)
    if base_name == 'cube':
        quat = generate_random_quaternion()
        return position + quat
    else:
        return position
    
def modify_dict(world_yml):
    # Open dictionary
    world_params = open_yaml(world_yml)
    # Select random indexes
    indexes_spheres = select_random_indexes(0, 10, 5)
    indexes_cubes = select_random_indexes(11, 35, 10)
    indexes = indexes_spheres + indexes_cubes
    #print(f"indexes: {indexes}")
    # Get objects from dictionary
    selected_objects = get_objects_by_indexes(world_params, indexes)
    #print(f"selected_objects: {selected_objects}")

    # Create new dictionary
    compressed_world_params = {'world_model': {'coll_objs': {'sphere': {},'cube': {}}}}

    sphere_index = 1
    cube_index = 1
    for i in range(len(indexes)):
        obj = selected_objects[i]
        name = obj[0]
        base_name = get_base_name(name)

        new_pos = randomize_pos(obj, base_name)

        if base_name == 'sphere':
            # Modify dict
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

    #print(f"modyfied dict: {world_params}")
    #print(f"new dict: {compressed_world_params}")

    return world_params, indexes, compressed_world_params

###############################################################################################
###############################################################################################

    
if __name__ == '__main__':
    
    # instantiate empty gym:
    parser = argparse.ArgumentParser(description='pass args')
    parser.add_argument('--robot', type=str, default='franka', help='Robot to spawn')
    parser.add_argument('--cuda', action='store_true', default=True, help='use cuda')
    parser.add_argument('--headless', action='store_true', default=False, help='headless gym')
    parser.add_argument('--control_space', type=str, default='acc', help='Robot to spawn')
    args = parser.parse_args()
    
    sim_params = load_yaml(join_path(get_gym_configs_path(),'physx.yml'))
    sim_params['headless'] = args.headless
    gym_instance = Gym(**sim_params)
    
    
    Mpc = MpcRobotInteractive(args, gym_instance)

    i = 0
    while(i > -100):
        cost_params = {"manipulability": i, "stop_cost": 50, "stop_cost_acc": i, "smooth": i/1000, "state_bound": i, "ee_vel": i, "robot_self_collision" : i, "primitive_collision" : i, 
                        "voxel_collision" : 0}
        mpc_params = {"horizon" : 20 , "particles" : 150 }
        arm_configuration, goal_pose, end_effector_pos, end_effector_quat, done = Mpc.step(cost_params, mpc_params, i)
        if i%500 == 0:
            Mpc.reset()
        if done:
            break
        i += 1
    
