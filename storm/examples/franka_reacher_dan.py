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
import math
import time
from isaacgym import gymapi
from isaacgym import gymutil
import torch
torch.multiprocessing.set_start_method('spawn',force=True)
torch.set_num_threads(8)
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt
import yaml
import argparse
import numpy as np
from quaternion import quaternion, from_rotation_vector, from_rotation_matrix
import matplotlib.pyplot as plt
from quaternion import from_euler_angles, as_float_array, as_rotation_matrix, from_float_array, as_quat_array
from storm_kit.gym.core import Gym, World
from storm_kit.gym.sim_robot import RobotSim
from storm_kit.util_file import get_configs_path, get_gym_configs_path, join_path, load_yaml, get_assets_path
from storm_kit.gym.helpers import load_struct_from_dict
from storm_kit.util_file import get_mpc_configs_path as mpc_configs_path
from storm_kit.differentiable_robot_model.coordinate_transform import quaternion_to_matrix, CoordinateTransform
from storm_kit.mpc.task.reacher_task import ReacherTask
np.set_printoptions(precision=5)
from BGU.Rlpt.DebugTools.storm_tools import RealWorldState
from BGU.Rlpt.DebugTools.logger_config import logger, logger_ticks
from BGU.Rlpt.DebugTools.globs import globs

import json

import threading
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

import threading
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
 
def goal_test(position_norm:float,orientation_norm:float, orientation_epsilon = 0.01, position_epsilon = 0.01) -> bool:
    reached_position = position_norm < position_epsilon
    reached_orientation = orientation_norm < orientation_epsilon
    return reached_position and reached_orientation 
    

def mpc_robot_interactive(args, gym_instance:Gym):

    vis_ee_target = True # Dan - what is this ?
    robot_file = args.robot + '.yml'
    task_file = args.robot + '_reacher.yml'
    world_file = 'collision_primitives_3d.yml'
    # world_file = 'collision_primitives_3d_dan.yml'
    
    gym = gym_instance.gym # Dan - what is this ? Gym object
    sim = gym_instance.sim # Dan - what is this ?
    world_yml = join_path(get_gym_configs_path(), world_file) # Dan - this is the location of static items (Cubes, balls)
    with open(world_yml) as file:
        world_params = yaml.load(file, Loader=yaml.FullLoader)
    robot_yml = join_path(get_gym_configs_path(),args.robot + '.yml')
    with open(robot_yml) as file:
        robot_params = yaml.load(file, Loader=yaml.FullLoader)
    sim_params = robot_params['sim_params']
    sim_params['asset_root'] = get_assets_path()
    
    if(args.cuda): # Dan - make sure we use cuda
        device = 'cuda'
        print("Using cuda")
    else:
        device = 'cpu'
    sim_params['collision_model'] = None # Dan - what is this ?
    
    
    # create robot simulation:
    robot_sim = RobotSim(gym_instance=gym, sim_instance=sim, **sim_params, device=device)

    
    # create gym environment:
    robot_pose = sim_params['robot_pose']
    env_ptr = gym_instance.env_list[0] 
    robot_ptr = robot_sim.spawn_robot(env_ptr, robot_pose, coll_id=2)

    device = torch.device('cuda', 0) 
    
    tensor_args = {'device':device, 'dtype':torch.float32}
    
    # Dan: tried to debug camera... failed
        # youtube: https://www.google.com/search?q=isaac-gym+rotate+camera&oq=isaac-gym+rotate+camera&gs_lcrp=EgZjaHJvbWUyBggAEEUYOTIHCAEQIRigAdIBCDU4ODJqMGo3qAIAsAIA&sourceid=chrome&ie=UTF-8#fpstate=ive&vld=cid:1201e946,vid:nleDq-oJjGk,st:0    
        # debug_camera = True
        # if debug_camera:                
        #     camera_props = gymapi.CameraProperties()
        #     camera_props.width = 128
        #     camera_props.height = 128
        #     camera_handle = gym.create_camera_sensor(env_ptr, camera_props)
        #     gym.set_camera_location(camera_handle, env_ptr, gymapi.Vec3(100,100,100), gymapi.Vec3(0,0,0))
        #     gym.render_all_camera_sensors(sim)
        
    # spawn camera:
    robot_camera_pose = np.array([1.6,-1.5, 1.8,0.707,0.0,0.0,0.707])
    q = as_float_array(from_euler_angles(-0.5 * 90.0 * 0.01745, 50.0 * 0.01745, 90 * 0.01745))
    robot_camera_pose[3:] = np.array([q[1], q[2], q[3], q[0]])
    robot_sim.spawn_camera(env_ptr, 60, 640, 480, robot_camera_pose)

        

    # get pose
    w_T_r = copy.deepcopy(robot_sim.spawn_robot_pose)
    
    w_T_robot = torch.eye(4)
    quat = torch.tensor([w_T_r.r.w,w_T_r.r.x,w_T_r.r.y,w_T_r.r.z]).unsqueeze(0)
    rot = quaternion_to_matrix(quat)
    w_T_robot[0,3] = w_T_r.p.x
    w_T_robot[1,3] = w_T_r.p.y
    w_T_robot[2,3] = w_T_r.p.z
    w_T_robot[:3,:3] = rot[0]
    world_instance = World(gym, sim, env_ptr, world_params, w_T_r=w_T_r)
    
    # >>>>>> Dan - we need those??? >>>>>>>>>>>>>>>>>>>
    table_dims = np.ravel([1.5,2.5,0.7])
    cube_pose = np.ravel([0.35, -0.0,-0.35,0.0, 0.0, 0.0,1.0])
    cube_pose = np.ravel([0.9,0.3,0.4, 0.0, 0.0, 0.0,1.0])
    table_dims = np.ravel([0.35,0.1,0.8])
    cube_pose = np.ravel([0.35,0.3,0.4, 0.0, 0.0, 0.0,1.0])
    table_dims = np.ravel([0.3,0.1,0.8])
    # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
    
    # get camera data:
    mpc_control = ReacherTask(task_file, robot_file, world_file, tensor_args) # Inherits from ArmTask which inherits from BaseTask
    n_dof = mpc_control.controller.rollout_fn.dynamics_model.n_dofs
    start_qdd = torch.zeros(n_dof, **tensor_args) # Dan - we need this??? 
    # update goal:
    exp_params = mpc_control.exp_params # Dan - we need this??? 
    current_state = copy.deepcopy(robot_sim.get_state(env_ptr, robot_ptr)) # Dan - we need this??? 
    ee_list = []# Dan - we need this??? 
    mpc_tensor_dtype = {'device':device, 'dtype':torch.float32}
    franka_bl_state = np.array([-0.3, 0.3, 0.2, -2.0, 0.0, 2.4,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0])
    x_des_list = [franka_bl_state]
    ee_error = 10.0 # Dan - we need this??? 
    j = 0 # ?
    t_step = 0 #  Dan - we need this???
    i = 0
    x_des = x_des_list[0]
    mpc_control.update_params(goal_state=x_des)
    # spawn object:
    x,y,z = 0.0, 0.0, 0.0
    tray_color = gymapi.Vec3(0.8, 0.1, 0.1)
    asset_options = gymapi.AssetOptions() # Dan: ?
    asset_options.armature = 0.001 # Dan: ?
    asset_options.fix_base_link = True # Dan: ?
    asset_options.thickness = 0.002 # Dan: ?
    object_pose = gymapi.Transform() # Dan: ?
    object_pose.p = gymapi.Vec3(x, y, z) # Dan: ?
    object_pose.r = gymapi.Quat(0,0,0, 1) # Dan: ?
    obj_asset_file = "urdf/mug/movable_mug.urdf" # Dan: ?
    obj_asset_root = get_assets_path()# Dan: ?
    if(vis_ee_target): # Dan: ?
        target_object = world_instance.spawn_object(obj_asset_file, obj_asset_root, object_pose, color=tray_color, name='ee_target_object') # Dan red (goal) mug
        obj_base_handle = gym.get_actor_rigid_body_handle(env_ptr, target_object, 0)
        obj_body_handle = gym.get_actor_rigid_body_handle(env_ptr, target_object, 6)
        gym.set_rigid_body_color(env_ptr, target_object, 0, gymapi.MESH_VISUAL_AND_COLLISION, tray_color)
        gym.set_rigid_body_color(env_ptr, target_object, 6, gymapi.MESH_VISUAL_AND_COLLISION, tray_color)
        obj_asset_file = "urdf/mug/mug.urdf"
        obj_asset_root = get_assets_path()
        ee_handle = world_instance.spawn_object(obj_asset_file, obj_asset_root, object_pose, color=tray_color, name='ee_current_as_mug') # Dan green (end effector) mug
        ee_body_handle = gym.get_actor_rigid_body_handle(env_ptr, ee_handle, 0)
        tray_color = gymapi.Vec3(0.0, 0.8, 0.0)
        gym.set_rigid_body_color(env_ptr, ee_handle, 0, gymapi.MESH_VISUAL_AND_COLLISION, tray_color)

    g_pos = np.ravel(mpc_control.controller.rollout_fn.goal_ee_pos.cpu().numpy()) # Dan: ?
    g_q = np.ravel(mpc_control.controller.rollout_fn.goal_ee_quat.cpu().numpy()) # Dan: ?
    object_pose.p = gymapi.Vec3(g_pos[0], g_pos[1], g_pos[2]) # Dan: ?
    object_pose.r = gymapi.Quat(g_q[1], g_q[2], g_q[3], g_q[0]) # Dan: ?
    object_pose = w_T_r * object_pose # Dan: ?
    if(vis_ee_target): # Dan: ?
        gym.set_rigid_transform(env_ptr, obj_base_handle, object_pose) # Dan: ?
    n_dof = mpc_control.controller.rollout_fn.dynamics_model.n_dofs # Dan: ?
    prev_acc = np.zeros(n_dof) # Dan: ?
    ee_pose = gymapi.Transform() # Dan: ?
    w_robot_coord = CoordinateTransform(trans=w_T_robot[0:3,3].unsqueeze(0), rot=w_T_robot[0:3,0:3].unsqueeze(0)) # Dan: ?
    rollout = mpc_control.controller.rollout_fn # Dan - we need this?
    tensor_args = mpc_tensor_dtype
    sim_dt = mpc_control.exp_params['control_dt']
    log_traj = {'q':[], 'q_des':[], 'qdd_des':[], 'qd_des':[],
                'qddd_des':[]} # Dan: we need this?

    q_des = None 
    qd_des = None # Dan - we need this?
    t_step = gym_instance.get_sim_time()

    g_pos = np.ravel(mpc_control.controller.rollout_fn.goal_ee_pos.cpu().numpy())
    g_q = np.ravel(mpc_control.controller.rollout_fn.goal_ee_quat.cpu().numpy())
    
    episode_parameters = {
        'horizon': mpc_control.controller.horizon,
        'num_particles': mpc_control.controller.num_particles,
        'weights': None
    } 
    
    episode_results = {
        'ee_dist_to_target_position': None, 
        'ee_dist_to_target_orientation': None,
        'ee_cputime_to_target': None,
        'ee_steps_to_target': None,
        'gym_simtime': None,
        'arm_min_dist_to_obj_along_traj': math.inf,
        'total_cost' : None,
        'smoothness': None
    }
    #all_weights = [] # over all steps in real world
    all_costs = []
    time_start = time.process_time()
    while(i > -100): # Dan - every iter makes a step in real world
        
        RealWorldState.reset(i)
        # if i % logger_ticks == 0:
        #     logger.debug(f'Real world time = {RealWorldState.real_world_time}')
        try:
            # >>>>>> Dan: REAL WORLD/GUI STEP >>>>>>>>
            gym_instance.step()
            #all_weights.append(RealWorldState.get_costs_weight_list(RealWorldState.filter_out_tensors(RealWorldState.cost)))
            
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            if(vis_ee_target): # Dan what is 'vis'?
                pose = copy.deepcopy(world_instance.get_pose(obj_body_handle))
                pose = copy.deepcopy(w_T_r.inverse() * pose)
                if(np.linalg.norm(g_pos - np.ravel([pose.p.x, pose.p.y, pose.p.z])) > 0.00001 or (np.linalg.norm(g_q - np.ravel([pose.r.w, pose.r.x, pose.r.y, pose.r.z]))>0.0)):
                    g_pos[0] = pose.p.x
                    g_pos[1] = pose.p.y
                    g_pos[2] = pose.p.z
                    g_q[1] = pose.r.x
                    g_q[2] = pose.r.y
                    g_q[3] = pose.r.z
                    g_q[0] = pose.r.w
                    mpc_control.update_params(goal_ee_pos=g_pos,
                                              goal_ee_quat=g_q)
            t_step += sim_dt
            current_robot_state = copy.deepcopy(robot_sim.get_state(env_ptr, robot_ptr))
            
            # >>>> Dan - get_command() - predict forward from previous action and previous state (?????) >>>>>>>>>>            
            
            
            ####################
            ####################
            # mpc_control.get_command running the mpc rollouts and then returns 
            # the command to the real contoller
            #####################
            #####################
            command = mpc_control.get_command(t_step, current_robot_state, control_dt=sim_dt, WAIT=True) 
            
            # command format = command.keys() = ['name', 'position', 'velocity', 'acceleration'])
            # example:
            # {'name': ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'], 'position': array([ 0.88,  0.14, -0.97, -2.04,  0.24,  2.09,  0.98], dtype=float32), 'velocity': array([-0.03,  0.58, -0.19,  0.15,  0.42,  0.09,  0.19], dtype=float32), 'acceleration': array([-0.13, -0.22,  0.25,  0.15, -0.01, -0.16,  0.02], dtype=float32)}
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

            # >>>>>>>>>>>>>>>> Dan - FETCH STATE OF JOINTS from cuurent_state  >>>>>>>>>>>>>>>>>>>>>>>>>>>>>
            # Dan: this is a dict of the joint state (position, velocity, acceleration)... They use a few objects for representing it
            filtered_state_mpc = current_robot_state #mpc_control.current_state - # {'name': ['panda_joint1', 'panda_joint2', 'panda_joint3', 'panda_joint4', 'panda_joint5', 'panda_joint6', 'panda_joint7'], 'position': array([ 0.88,  0.07, -0.95, -2.06,  0.2 ,  2.08,  0.97], dtype=float32), 'velocity': array([ 0.  ,  0.6 , -0.21,  0.12,  0.42,  0.12,  0.25], dtype=float32), 'acceleration': array([ 0.,  0., -0.,  0.,  0.,  0.,  0.], dtype=float32)}
            # Dan: some cosmetic changes in state...
            curr_state = np.hstack((filtered_state_mpc['position'], filtered_state_mpc['velocity'], filtered_state_mpc['acceleration'])) 
            curr_state_tensor = torch.as_tensor(curr_state, **tensor_args).unsqueeze(0)    
                
            # >>>>>>>>>>>>>>>> Dan - keeping only the position part of the command >>>>>>>>>
            q_des = copy.deepcopy(command['position']) # Dan: command['position']-  WILL BE THE COMMAND SENT TO THE ROBOT    
            # qd_des = copy.deepcopy(command['velocity']) #* 0.5 # Dan - we need it?
            # qdd_des = copy.deepcopy(command['acceleration']) # Dan - we need it?
        
            # ???????????????????? Dan - get end effector current ERROR (we need it?) ????????????????????    
            # Dan: BaseTask/get_current_error() -> ArmBase.current_cost()->cost_fn() -> returns end effector error            
            ee_error = mpc_control.get_current_error(filtered_state_mpc) 
            # ee_error[0] = tensor([[768.1894]], device='cuda:0', dtype=torch.float32)            
            # ee_error[1] =tensor([[1.3212]], device='cuda:0', dtype=torch.float32)
            # ee_error[2] = tensor([[[0.7462]]], device='cuda:0', dtype=torch.float32
            # ?????????????????????????????????????????????????????????????????????????????????????????????
            
            # >>>>> Dan - get end effector current POSITION & ORIENTATION >>>>>
            pose_state = mpc_control.controller.rollout_fn.get_ee_pose(curr_state_tensor)
            # get current pose:
            e_pos = np.ravel(pose_state['ee_pos_seq'].cpu().numpy())
            e_quat = np.ravel(pose_state['ee_quat_seq'].cpu().numpy())
            ee_pose.p = copy.deepcopy(gymapi.Vec3(e_pos[0], e_pos[1], e_pos[2])) # Dan p = position
            ee_pose.r = gymapi.Quat(e_quat[1], e_quat[2], e_quat[3], e_quat[0]) # Dan r = rotation (quaternion)
            ee_pose = copy.deepcopy(w_T_r) * copy.deepcopy(ee_pose)
            
            if(vis_ee_target): # Dan ?
                gym.set_rigid_transform(env_ptr, ee_body_handle, copy.deepcopy(ee_pose))
            
            # >>>>> Dan - end effector distance to target calculation >>>>>>            
            # >>>>just renaming for better comprehensiveness >>>
            target_position = g_pos # target end effector location(x,y,z) 
            current_position =  e_pos # current end effector location(x,y,z)
            target_orientation = g_q # target end effector orientation/rotation (q0,q1,q2,q3)
            current_orientation =  e_quat # current end effector orientation/rotation(q0,q1,q2,q3)
    
            # >>>>> Dan - calculate distances to target - orientation and position >>>>>>        
            vdiff_position = target_position - current_position 
            vdiff_orientation = target_orientation - current_orientation 
            vdiff_position_norm = np.linalg.norm(vdiff_position) 
            vdiff_orientation_norm = np.linalg.norm(vdiff_orientation)
            
            cost_current_state = RealWorldState.cost
            clean_view_cost_current_state = RealWorldState.clean_view(cost_current_state) 
            
            # clean_view_cost_current_state_weights_only = RealWorldState.get_costs_weight_list(clean_view_cost_current_state)
            # all_weights.append(clean_view_cost_current_state_weights_only)
            all_costs.append(clean_view_cost_current_state)
            if RealWorldState.real_world_time == 0:
                episode_parameters['weights'] = RealWorldState.get_costs_weight_list(clean_view_cost_current_state)
            
            # >>>>>>>>>>> Dan - goal test - reached target? >>>>>>>>>>>.
            if goal_test(vdiff_position_norm, vdiff_orientation_norm): 
                logger.info(f'----------------------------SIMULATION IS OVER----------------------------')
                logger.info(f'simulation input parameters:\n\
                    {json.dumps(episode_parameters, indent=4)}\n')
                episode_results['ee_dist_to_target_position'] = str(vdiff_position_norm)
                episode_results['ee_dist_to_target_orientation'] = str(vdiff_orientation_norm)
                episode_results['ee_cputime_to_target'] = str(time.process_time() - time_start)
                episode_results['ee_steps_to_target'] =  RealWorldState.real_world_time        
                episode_results['gym_simtime'] =  gym_instance.get_sim_time()        
                episode_results['total_cost'] =  RealWorldState.total_cost              
                logger.info(f'simulation results:\n{json.dumps(episode_results, indent=4)}')

                ##### exit #####
                timeout = 10
                print(f'Leaving simulation in {timeout} seconds')
                for i in range(timeout, -1, -1):
                    print(i)
                    time.sleep(1)
                break
            
            # >>>>>> Dan: print every logger_ticks time-units >>>>>>>>>>>>>>
            if i % logger_ticks == 0:
                logger.debug(f'-------------{i} steps in real world have passed, sim time gym = {gym_instance.get_sim_time()} -------------')
                logger.debug(f'target position (red mug): {target_position}')
                logger.debug(f'current position (green mug): {current_position}')
                logger.debug(f'target orientation (red mug): {target_orientation}')
                logger.debug(f'current orientation (green mug) : {current_orientation}')    
                logger.debug(f'target position  - current position =  {vdiff_position}')
                logger.debug(f'distance to target position (l2 norm) = {vdiff_position_norm} ') 
                logger.debug(f'target orientaion - current orientaion = {vdiff_orientation}')
                logger.debug(f'distance to target orientation (l2 norm) = {vdiff_orientation_norm}\n') 
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<,<<<<<
        
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
            
            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
            robot_sim.command_robot_position(q_des, env_ptr, robot_ptr) # Dan - RobotSim..gym.set_actor_dof_position_targets()
            #robot_sim.set_robot_state(q_des, qd_des, env_ptr, robot_ptr)
            current_state = command # Dan - we need it?

            
            i += 1 #Dan update time
            
            
            
        except KeyboardInterrupt:
            print('Closing')
            done = True
            break
    mpc_control.close()
    return 1 
    
if __name__ == '__main__':
    
    # instantiate empty gym:
    logger.info('parsing arguments...\n')
    parser = argparse.ArgumentParser(description='pass args')
    parser.add_argument('--robot', type=str, default='franka', help='Robot to spawn')
    parser.add_argument('--cuda', action='store_true', default=True, help='use cuda')
    parser.add_argument('--headless', action='store_true', default=False, help='headless gym')
    parser.add_argument('--control_space', type=str, default='acc', help='Robot to spawn')
    args = parser.parse_args()
    logger.info(f'terminal user arguments: {args}\n')
    sim_params = load_yaml(join_path(get_gym_configs_path(),'physx.yml'))
    logger.info(f'simulation params: {sim_params}\n')
    
    sim_params['headless'] = args.headless
    logger.info('starting gym gui...\n')
    gym_instance = Gym(**sim_params) # Dan - starting the gym gui window
    logger.info('starting simulation...\n')
    
    mpc_robot_interactive(args, gym_instance)

