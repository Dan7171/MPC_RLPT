
'''
HER paper:
https://proceedings.neurips.cc/paper_files/paper/2017/file/453fadbd8a1a3af50a9df4df899537b5-Paper.pdf

'''

import random

import numpy as np
import torch
from BGU.Rlpt import rlpt_agent
from BGU.Rlpt.utils.error import pos_error, pose_as_ndarray, rot_error
from BGU.Rlpt.utils.type_operations import as_1d_tensor, as_2d_tensor

class HindsightExperienceReplay: # HER 
    def __init__(self, cfg):
        self.episode_transitions = []
        self._episode_transitions_info_for_reward_computation = []
        self.strategy = cfg['strategy'] if 'strategy' in cfg else 'future' 
        self.N = cfg['N']
        self.k = cfg['k']
        
    def add_transition(self, transition:tuple, info_for_reward_computation:dict):
        self.episode_transitions.append(transition)
        self._episode_transitions_info_for_reward_computation.append(info_for_reward_computation)

    def _sample_additional_goals(self, ts, strategy, k=8):
            """
            
            strategy:
            final: additional goals we use for replay are the ones corresponding to the final state of the environment. 
            future: (most recommended (try k=4/8)). Replay with k random states which come from the same episode as the transition being replayed and were observed after it.
            
            """
            G = []
            
            if strategy == 'future':
                next_transitions_in_episode_range = range(ts, len(self.episode_transitions)) # from each transition, we take the "next state" (the state it was reached to). from s(ts+1) inclusive to s(final) inclusive) 
                k = min(k,len(next_transitions_in_episode_range))
                G = [None] * k
                sampled_next_transitions_timesteps = random.sample(next_transitions_in_episode_range, k) # get k "next states" like
                # print(f"debug {k} new goals to add in buffer for update {ts}, indices = {sampled_next_transitions_timesteps}")
                
                for i in range(k):
                    sampled_transition_ts = sampled_next_transitions_timesteps[i] # sampled transition (identified by its time step)
                    new_goal_pose_gym = self._episode_transitions_info_for_reward_computation[sampled_transition_ts]['s_next_ee_pose_gym'] # this is the representation which is relevant for the reward computation
                    G[i] = new_goal_pose_gym # set ith "next state" as a goal in the additional goals list     
            
            return G
    
    def _make_modified_state_copy_with_new_goal(self, rlpt_agent, st_tensor, new_goal)-> np.ndarray:
        return rlpt_agent.make_modified_state(st_tensor, 'goal_pose',new_goal)  
        
    def _compute_reward_for_new_goal(self,rlpt_agent, new_goal, transition_info):
        s_next_ee_pose_gym = transition_info['s_next_ee_pose_gym'] # pose where actually_reached
        s_next_ee_pos_error_wr_to_new_goal = pos_error(s_next_ee_pose_gym.p, new_goal.p) # end effector position error (s(t+1))
        s_next_ee_rot_error_wr_to_new_goal = rot_error(s_next_ee_pose_gym.r, new_goal.r)  # end effector rotation error (s(t+1))   
        s_next_contact_detected = transition_info['s_next_contact_detected']
        step_duration = transition_info['step_duration']
        return rlpt_agent.compute_reward(s_next_ee_pos_error_wr_to_new_goal, s_next_ee_rot_error_wr_to_new_goal,s_next_contact_detected, step_duration)
                
    def optimize(self, rlpt_agent):
        T = len(self.episode_transitions)
        N = self.N if self.N != -1 else T  # num of optimization steps
        for t in range(T):
            G = self._sample_additional_goals(t, strategy=self.strategy, k=self.k) 
            st_tensor, at_idx_tensor, s_next_tensor = self.episode_transitions[t]
            transition_info = self._episode_transitions_info_for_reward_computation[t]
            
            for g_tag in G:
                g_tag_np_flatten = pose_as_ndarray(g_tag).flatten()  
                st_with_g_tag = self._make_modified_state_copy_with_new_goal(rlpt_agent, st_tensor, g_tag_np_flatten)
                st_next_with_g_tag = self._make_modified_state_copy_with_new_goal(rlpt_agent, s_next_tensor, g_tag_np_flatten)
                r_tag = self._compute_reward_for_new_goal(rlpt_agent, g_tag, transition_info) 
                rlpt_agent.train_suit.memory.push(st_with_g_tag, at_idx_tensor, st_next_with_g_tag, as_1d_tensor([r_tag])) 
                # print("debug g tag")
                # print(g_tag_np_flatten)
                
        for t in range(N):
            optim_meta_data = rlpt_agent.optimize() # TODO: Should make the C of fixed targets update support HER too 
            print('debug: optim meta data of HER updates')
            print(optim_meta_data)
        return optim_meta_data