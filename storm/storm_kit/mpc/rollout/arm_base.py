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
import time
import torch
import torch.autograd.profiler as profiler

from ..cost import DistCost, PoseCost, ProjectedDistCost, JacobianCost, ZeroCost, EEVelCost, StopCost, FiniteDifferenceCost
from ..cost.bound_cost import BoundCost
from ..cost.manipulability_cost import ManipulabilityCost
from ..cost import CollisionCost, VoxelCollisionCost, PrimitiveCollisionCost
from ..model import URDFKinematicModel
from ...util_file import join_path, get_assets_path
from ...differentiable_robot_model.coordinate_transform import matrix_to_quaternion, quaternion_to_matrix
from ...mpc.model.integration_utils import build_fd_matrix
from ...mpc.rollout.rollout_base import RolloutBase
from ..cost.robot_self_collision_cost import RobotSelfCollisionCost
from BGU.Rlpt.DebugTools.storm_tools import RealWorldState, is_real_world 
from BGU.Rlpt.DebugTools.logger_config import logger


class ArmBase(RolloutBase):
    """
    This rollout function is for reaching a cartesian pose for a robot

    Todo: 
    1. Update exp_params to be kwargs
    """

    def __init__(self, exp_params, tensor_args={'device':"cpu", 'dtype':torch.float32}, world_params=None):
        self.tensor_args = tensor_args
        self.exp_params = exp_params
        mppi_params = exp_params['mppi']
        model_params = exp_params['model']

        robot_params = exp_params['robot_params']
        
        assets_path = get_assets_path()
        #print('EE LINK',exp_params['model']['ee_link_name'])
        # initialize dynamics model:
        dynamics_horizon = mppi_params['horizon'] * model_params['dt']
        #Create the dynamical system used for rollouts
        self.dynamics_model = URDFKinematicModel(join_path(assets_path,exp_params['model']['urdf_path']),
                                                 dt=exp_params['model']['dt'],
                                                 batch_size=mppi_params['num_particles'],
                                                 horizon=dynamics_horizon,
                                                 tensor_args=self.tensor_args,
                                                 ee_link_name=exp_params['model']['ee_link_name'],
                                                 link_names=exp_params['model']['link_names'],
                                                 dt_traj_params=exp_params['model']['dt_traj_params'],
                                                 control_space=exp_params['control_space'],
                                                 vel_scale=exp_params['model']['vel_scale'])
        self.dt = self.dynamics_model.dt
        self.n_dofs = self.dynamics_model.n_dofs
        # rollout traj_dt starts from dt->dt*(horizon+1) as tstep 0 is the current state
        #self.traj_dt = torch.arange(self.dt, (mppi_params['horizon'] + 1) * self.dt, self.dt, device=device, dtype=float_dtype)
        self.traj_dt = self.dynamics_model.traj_dt
        
        self.fd_matrix = build_fd_matrix(10 - self.exp_params['cost']['smooth']['order'], device=self.tensor_args['device'], dtype=self.tensor_args['dtype'], PREV_STATE=True, order=self.exp_params['cost']['smooth']['order'])
        self.goal_state = None
        self.goal_ee_pos = None
        self.goal_ee_rot = None
        
        device = self.tensor_args['device']
        float_dtype = self.tensor_args['dtype']

        # Cost terms initiation:
        
        self.jacobian_cost = JacobianCost(ndofs=self.n_dofs, device=device, float_dtype=float_dtype, retract_weight=exp_params['cost']['retract_weight'])
        self.null_cost = ProjectedDistCost(ndofs=self.n_dofs, device=device, float_dtype=float_dtype, **exp_params['cost']['null_space'])
        self.manipulability_cost = ManipulabilityCost(ndofs=self.n_dofs, device=device, float_dtype=float_dtype, **exp_params['cost']['manipulability'])
        self.zero_vel_cost = ZeroCost(device=device, float_dtype=float_dtype, **exp_params['cost']['zero_vel'])
        self.zero_acc_cost = ZeroCost(device=device, float_dtype=float_dtype, **exp_params['cost']['zero_acc'])
        self.stop_cost = StopCost(**exp_params['cost']['stop_cost'], tensor_args=self.tensor_args, traj_dt=self.traj_dt)
        self.stop_cost_acc = StopCost(**exp_params['cost']['stop_cost_acc'], tensor_args=self.tensor_args, traj_dt=self.traj_dt)
        self.retract_state = torch.tensor([self.exp_params['cost']['retract_state']], device=device, dtype=float_dtype)
        
        if self.exp_params['cost']['smooth']['weight'] > 0:
            self.smooth_cost = FiniteDifferenceCost(**self.exp_params['cost']['smooth'], tensor_args=self.tensor_args)
        if self.exp_params['cost']['voxel_collision']['weight'] > 0:
            self.voxel_collision_cost = VoxelCollisionCost(robot_params=robot_params,tensor_args=self.tensor_args,**self.exp_params['cost']['voxel_collision'])
        if exp_params['cost']['primitive_collision']['weight'] > 0.0:
            self.primitive_collision_cost = PrimitiveCollisionCost(world_params=world_params, robot_params=robot_params, tensor_args=self.tensor_args, **self.exp_params['cost']['primitive_collision'])
        if exp_params['cost']['robot_self_collision']['weight'] > 0.0:
            self.robot_self_collision_cost = RobotSelfCollisionCost(robot_params=robot_params, tensor_args=self.tensor_args, **self.exp_params['cost']['robot_self_collision'])
            
        self.ee_vel_cost = EEVelCost(ndofs=self.n_dofs,device=device, float_dtype=float_dtype,**exp_params['cost']['ee_vel'])
        bounds = torch.cat([self.dynamics_model.state_lower_bounds[:self.n_dofs * 3].unsqueeze(0),self.dynamics_model.state_upper_bounds[:self.n_dofs * 3].unsqueeze(0)], dim=0).T
        self.bound_cost = BoundCost(**exp_params['cost']['state_bound'],tensor_args=self.tensor_args,bounds=bounds)
        
        self.link_pos_seq = torch.zeros((1, 1, len(self.dynamics_model.link_names), 3), **self.tensor_args)
        self.link_rot_seq = torch.zeros((1, 1, len(self.dynamics_model.link_names), 3, 3), **self.tensor_args)
        
        self.prev_ts_cost_weights = None 
        self.world_params = None     
    
    def cost_fn(self, state_dict, action_batch, no_coll=False, horizon_cost=True):
        """
        Calculting costs
        
        TODO: My assumption is that when called with horizon_cost=True and no_coll=False, it relates to the cost calculation
        at the rollouts (planning) and when its False, it is when calculating the cost at real world (as a result of step at one time step)
        """
        
            
        ee_pos_batch, ee_rot_batch = state_dict['ee_pos_seq'], state_dict['ee_rot_seq']
        state_batch = state_dict['state_seq']
        lin_jac_batch, ang_jac_batch = state_dict['lin_jac_seq'], state_dict['ang_jac_seq']
        link_pos_batch, link_rot_batch = state_dict['link_pos_seq'], state_dict['link_rot_seq']
        prev_state = state_dict['prev_state_seq']
        prev_state_tstep = state_dict['prev_state_seq'][:,-1]
        
        
        retract_state = self.retract_state
        
        J_full = torch.cat((lin_jac_batch, ang_jac_batch), dim=-2)
        
        # Null disp cost (ProjectedDistCost, Null space cost)
        null_disp_cost = self.null_cost.forward(state_batch[:,:,0:self.n_dofs] -
                                                retract_state[:,0:self.n_dofs],
                                                J_full,
                                                proj_type='identity',
                                                dist_type='squared_l2')
        
        cost = None
        new_cost = null_disp_cost 
        cost = new_cost
        if(no_coll == True and horizon_cost == False):
            return cost
        
        # Manipulability cost
        if(self.exp_params['cost']['manipulability']['weight'] > 0.0):            
            new_cost = self.manipulability_cost.forward(J_full) 
            cost += new_cost
            
        if horizon_cost: # TODO: I think it is true when we are at the planning part (not real world) but need to verify
            
            # Stop cost
            if self.exp_params['cost']['stop_cost']['weight'] > 0:
                new_cost = self.stop_cost.forward(state_batch[:, :, self.n_dofs:self.n_dofs * 2]) 
                cost += new_cost
                
            # Stop cost acc (accelaration? accuracy?)
            if self.exp_params['cost']['stop_cost_acc']['weight'] > 0:
                new_cost = self.stop_cost_acc.forward(state_batch[:, :, self.n_dofs*2 :self.n_dofs * 3], is_stop_acc=True)
                cost += new_cost # Dan - it goes to the same place as stop goes,so I added that to distinguish

            # Smoothness cost
            if self.exp_params['cost']['smooth']['weight'] > 0:
                order = self.exp_params['cost']['smooth']['order']
                prev_dt = (self.fd_matrix @ prev_state_tstep)[-order:]
                n_mul = 1
                state = state_batch[:,:, self.n_dofs * n_mul:self.n_dofs * (n_mul+1)]
                p_state = prev_state[-order:,self.n_dofs * n_mul: self.n_dofs * (n_mul+1)].unsqueeze(0)
                p_state = p_state.expand(state.shape[0], -1, -1)
                state_buffer = torch.cat((p_state, state), dim=1)
                traj_dt = torch.cat((prev_dt, self.traj_dt))                
                new_cost = self.smooth_cost.forward(state_buffer, traj_dt) 
                cost += new_cost

        # State bound cost
        if self.exp_params['cost']['state_bound']['weight'] > 0:  
            new_cost = self.bound_cost.forward(state_batch[:,:,:self.n_dofs * 3]) 
            cost += new_cost
        
        # End effector velocity cost
        if self.exp_params['cost']['ee_vel']['weight'] > 0:
            new_cost = self.ee_vel_cost.forward(state_batch, lin_jac_batch)
            cost += new_cost
            
        if not no_coll :
            if self.exp_params['cost']['robot_self_collision']['weight'] > 0:    
                new_cost = self.robot_self_collision_cost.forward(state_batch[:,:,:self.n_dofs])
                cost += new_cost
            
            if self.exp_params['cost']['primitive_collision']['weight'] > 0:
                # logger.debug(f"primitive_collision weight = {self.exp_params['cost']['primitive_collision']['weight']} > 0 => cost += new cost")
                
                new_cost = self.primitive_collision_cost.forward(link_pos_batch, link_rot_batch)
                # logger.debug(f'primitive collision cost = \n{new_cost}\nChanged total cost? {bool((new_cost != 0).any())}')
                cost += new_cost
                
            if self.exp_params['cost']['voxel_collision']['weight'] > 0:
                # logger.debug(f"voxel_collision weight = {self.exp_params['cost']['voxel_collision']['weight']} > 0 => cost += new cost")
                # C
                new_cost = self.voxel_collision_cost.forward(link_pos_batch, link_rot_batch)
                # logger.debug(f'voxel collision cost = \n{new_cost}\nChanged total cost? {bool((new_cost != 0).any())}')
                cost += new_cost

        # logger.debug(f'exp params:{self.exp_params['cost']}')
        return cost
    
    def rollout_fn(self, start_state, act_seq):
        """
        Return sequence of costs and states encountered
        by simulating a batch of action sequences

        Parameters
        ----------
        action_seq: torch.Tensor [num_particles, horizon, d_act]
        """
        # rollout_start_time = time.time()
        #print("computing rollout")
        #print(act_seq)
        #print('step...')
        with profiler.record_function("robot_model"):
            state_dict = self.dynamics_model.rollout_open_loop(start_state, act_seq)
        
        #link_pos_seq, link_rot_seq = self.dynamics_model.get_link_poses()
        with profiler.record_function("cost_fns"):
            cost_seq = self.cost_fn(state_dict, act_seq)

        sim_trajs = dict(
            actions=act_seq,#.clone(),
            costs=cost_seq,#clone(),
            ee_pos_seq=state_dict['ee_pos_seq'],#.clone(),
            #link_pos_seq=link_pos_seq,
            #link_rot_seq=link_rot_seq,
            rollout_time=0.0
        )
        # Dam
        return sim_trajs

    def update_params(self, retract_state=None):
        """
        Updates the goal targets for the cost functions.

        """
        
        if(retract_state is not None):
            self.retract_state = torch.as_tensor(retract_state, **self.tensor_args).unsqueeze(0)
        
        return True
    
    def __call__(self, start_state, act_seq):
        """
        Will be called when running ArmBase with params (calling rollout_fn)
        """
        return self.rollout_fn(start_state, act_seq)
    
    def get_ee_pose(self, current_state):
        """
        Given current state of dofs of actor (for example franka), calculacting the end effector's current pose (current ee state). 
        
        Args:
            current_state: current dofs state

        Returns:
             current ee state (pose) in storm coordinate system
        """
        current_state = current_state.to(**self.tensor_args)
         
        
        ee_pos_batch, ee_rot_batch, lin_jac_batch, ang_jac_batch = self.dynamics_model.robot_model.compute_fk_and_jacobian(current_state[:,:self.dynamics_model.n_dofs], current_state[:, self.dynamics_model.n_dofs: self.dynamics_model.n_dofs * 2], self.exp_params['model']['ee_link_name'])

        ee_quat = matrix_to_quaternion(ee_rot_batch)
        state = {'ee_pos_seq':ee_pos_batch, 'ee_rot_seq':ee_rot_batch,
                 'lin_jac_seq': lin_jac_batch, 'ang_jac_seq': ang_jac_batch,
                 'ee_quat_seq':ee_quat}
        return state
    
    def current_cost(self, current_state, no_coll=True):
        """
        TODO : I belive this one being called when wanting to calculate the cost at simulator after each step (not in planning/rollouts) but need to verify
        """
        current_state = current_state.to(**self.tensor_args)
        
        curr_batch_size = 1
        num_traj_points = 1 #self.dynamics_model.num_traj_points
        
        ee_pos_batch, ee_rot_batch, lin_jac_batch, ang_jac_batch = self.dynamics_model.robot_model.compute_fk_and_jacobian(current_state[:,:self.dynamics_model.n_dofs], current_state[:, self.dynamics_model.n_dofs: self.dynamics_model.n_dofs * 2], self.exp_params['model']['ee_link_name'])


        link_pos_seq = self.link_pos_seq
        
        link_rot_seq = self.link_rot_seq

        # get link poses:
        for ki,k in enumerate(self.dynamics_model.link_names):
            link_pos, link_rot = self.dynamics_model.robot_model.get_link_pose(k)
            link_pos_seq[:,:,ki,:] = link_pos.view((curr_batch_size, num_traj_points,3))
            link_rot_seq[:,:,ki,:,:] = link_rot.view((curr_batch_size, num_traj_points,3,3))
            
        if(len(current_state.shape) == 2):
            current_state = current_state.unsqueeze(0)
            ee_pos_batch = ee_pos_batch.unsqueeze(0)
            ee_rot_batch = ee_rot_batch.unsqueeze(0)
            lin_jac_batch = lin_jac_batch.unsqueeze(0)
            ang_jac_batch = ang_jac_batch.unsqueeze(0)

        state_dict = {'ee_pos_seq':ee_pos_batch, 'ee_rot_seq':ee_rot_batch,
                      'lin_jac_seq': lin_jac_batch, 'ang_jac_seq': ang_jac_batch,
                      'state_seq': current_state,'link_pos_seq':link_pos_seq,
                      'link_rot_seq':link_rot_seq,
                      'prev_state_seq':current_state}
        
        cost = self.cost_fn(state_dict, None,no_coll=no_coll, horizon_cost=False, return_dist=True)
        
        return cost, state_dict
    
    #def update_costs(self, manipulability, stop_cost, stop_cost_acc, smooth, state_bound, ee_vel, robot_self_collision, primitive_collision, voxel_collision):
    def update_costs(self, new_params):
        """
        Setting new cost weights and other params to the cost terms
        """
        if "null_space" in new_params:        
            self.null_cost.update_weight(new_params["null_space"])
        if "manipulability" in new_params:
            self.manipulability_cost.update_weight(new_params["manipulability"])
        if "stop_cost" in new_params:
            self.stop_cost.update_all(new_params["stop_cost"])
        if "stop_cost_acc" in new_params:
            self.stop_cost_acc.update_all(new_params["stop_cost_acc"],is_stop_acc=True)
            
        
        if self.prev_ts_cost_weights is not None:  # when they are 0 , cost term is not used at all 
            if "smooth" in self.prev_ts_cost_weights and self.prev_ts_cost_weights["smooth"] > 0.0:
                self.smooth_cost.update_weight(self.prev_ts_cost_weights["smooth"])
            if "state_bound" in self.prev_ts_cost_weights and self.prev_ts_cost_weights["state_bound"] > 0.0:
                self.bound_cost.update_weight(self.prev_ts_cost_weights["state_bound"])
            if "ee_vel" in self.prev_ts_cost_weights and self.prev_ts_cost_weights["ee_vel"] > 0.0:
                self.ee_vel_cost.update_weight(self.prev_ts_cost_weights["ee_vel"]) 
            if "robot_self_collision" in self.prev_ts_cost_weights and self.prev_ts_cost_weights["robot_self_collision"] > 0.0:
                self.robot_self_collision_cost.update_weight(self.prev_ts_cost_weights["robot_self_collision"])
            if "primitive_collision" in self.prev_ts_cost_weights and self.prev_ts_cost_weights["primitive_collision"] > 0.0:
                self.primitive_collision_cost.update_weight(self.prev_ts_cost_weights["primitive_collision"])
            if "voxel_collision" in self.prev_ts_cost_weights and self.prev_ts_cost_weights["voxel_collision"] > 0.0:
                self.voxel_collision_cost.update_weight(self.prev_ts_cost_weights["voxel_collision"])
            
        self.prev_ts_cost_weights = new_params

    def update_mpc_params(self, mpc_params):
        self.dynamics_model.update_mpc_params(mpc_params)
        self.traj_dt = self.dynamics_model.traj_dt # torch vector, fixed for every H. Change as a result of H
        # these two are affected by self.traj_dt 
        self.stop_cost.update_max_vel(self.traj_dt)
        self.stop_cost_acc.update_max_vel(self.traj_dt)
        
    def update_world_params(self, world_params):
        """
        update positions of objects
        """
        self.world_params = world_params
        self.primitive_collision_cost.update_world_params(self.world_params, self.exp_params['robot_params'])
