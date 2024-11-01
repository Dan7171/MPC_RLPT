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
import torch
import torch.nn as nn

from BGU.Rlpt.DebugTools.storm_tools import RealWorldState, is_real_world
# import torch.nn.functional as F
from ...geom.sdf.robot_world import RobotWorldCollisionPrimitive
from .gaussian_projection import GaussianProjection
from BGU.Rlpt.Classes.CostTerm import CostTerm
from BGU.Rlpt.DebugTools.globs import GLobalVars
class PrimitiveCollisionCost(nn.Module):
    def __init__(self, weight=None, world_params=None, robot_params=None, gaussian_params={},
                 distance_threshold=0.1, tensor_args={'device':torch.device('cpu'), 'dtype':torch.float32}):
        super(PrimitiveCollisionCost, self).__init__()
        
        self.tensor_args = tensor_args
        self.weight = torch.as_tensor(weight,**self.tensor_args)
        
        self.proj_gaussian = GaussianProjection(gaussian_params=gaussian_params)

        robot_collision_params = robot_params['robot_collision_params']
        self.batch_size = -1
        # BUILD world and robot:
        self.robot_world_coll = RobotWorldCollisionPrimitive(robot_collision_params,
                                                             world_params['world_model'],
                                                             tensor_args=self.tensor_args,
                                                             bounds=robot_params['world_collision_params']['bounds'],
                                                             grid_resolution=robot_params['world_collision_params']['grid_resolution'])
        
        self.n_world_objs = self.robot_world_coll.world_coll.n_objs
        self.t_mat = None
        self.distance_threshold = distance_threshold
    def forward(self, link_pos_seq, link_rot_seq):

        
        inp_device = link_pos_seq.device
        batch_size = link_pos_seq.shape[0]
        horizon = link_pos_seq.shape[1]
        n_links = link_pos_seq.shape[2]

        if(self.batch_size != batch_size):
            self.batch_size = batch_size
            self.robot_world_coll.build_batch_features(self.batch_size * horizon, clone_pose=True, clone_points=True)

        link_pos_batch = link_pos_seq.view(batch_size * horizon, n_links, 3)
        link_rot_batch = link_rot_seq.view(batch_size * horizon, n_links, 3, 3)
        dist = self.robot_world_coll.check_robot_sphere_collisions(link_pos_batch, link_rot_batch)
        # signed distance (negative)
        dist = dist.view(batch_size, horizon, n_links)#, self.n_world_objs) 
        #print(f"all dists are negative dists: {len(dist[dist > 0]) < 1}")
        
        # safety_dist_debug = -0.005
        # if dist.shape == torch.Size([1,1,6]) and torch.any(dist > safety_dist_debug): # real world and collision - print red 
        #     print(f"'\033[91m'DEBUG!!!\
        #           \nmaximun penetration distance - robot links as spheres\
        #           \n{dist}\
        #          \n'\033[0m'")
            
        sniffer = GLobalVars.cost_sniffer
        safety_distance_rltp = 0.005 # alternative to distance_threshold
        penetration_to_obstacles_depth = dist + safety_distance_rltp
        if dist.shape == torch.Size([1,1,6]): # real world
            contact = torch.any(penetration_to_obstacles_depth > 0)
            # if sniffer.is_initialized():
            sniffer.is_contact_real_world = True if contact else False # contact detected
                
            if contact: # real world and collision - print red 
                print(f"'\033[91m'Collision detected!!!\
                    \nmaximum penetration depth into obstacles, by link index:\
                    \n{penetration_to_obstacles_depth}\
                        \n'\033[0m'")
                
                
        # cost only when dist is less
        dist += self.distance_threshold # like threshold (in cms) - distance (in cms). If > 0, distance to obstacle is smaller than allowed threshold.
        
            
        dist[dist <= 0.0] = 0.0 
        dist[dist > 0.2] = 0.2
        dist = dist / 0.25
        
        # Prints in red when real world premitive collision cost  > 0
        # if dist.shape == torch.Size([1,1,6]) and torch.any(dist > 0): # real world and collision - print red 
        #     print(f"'\033[91m'TOO CLOSE TO OBSTACLES!!!\
        #           \nmaximun penetration distance - robot links as spheres\
        #           \n{dist}\
        #          \n'\033[0m'")
            
        cost = torch.sum(dist, dim=-1)


        # cost = self.weight * cost 
        w1 = self.weight  
        t1 = cost  
        cost = w1 * t1  
        
        # sniffer = GLobalVars.cost_sniffer
        # if sniffer.is_initialized():
        sniffer.set('primitive_collision', CostTerm(w1, t1))
            
        return cost.to(inp_device)
    def update_weight(self, weight):
        """
        Update weight dynamically
        """
        self.weight = torch.as_tensor(weight, **self.tensor_args)
    def update_world_params(self, world_params, robot_params):
        """
        Update objects positions
        """
        self.robot_world_coll = RobotWorldCollisionPrimitive(robot_params['robot_collision_params'],
                                                             world_params['world_model'],
                                                             tensor_args=self.tensor_args,
                                                             bounds=robot_params['world_collision_params']['bounds'],
                                                             grid_resolution=robot_params['world_collision_params']['grid_resolution'])
        self.n_world_objs = self.robot_world_coll.world_coll.n_objs