
'''
HER paper:
https://proceedings.neurips.cc/paper_files/paper/2017/file/453fadbd8a1a3af50a9df4df899537b5-Paper.pdf

'''

import random

import numpy as np
import torch
from BGU.Rlpt import rlpt_agent
from BGU.Rlpt.franka_reacher_rlpt import goal_test
from BGU.Rlpt.utils.error import pos_error, pose_as_ndarray, rot_error
from BGU.Rlpt.utils.type_operations import as_1d_tensor, as_2d_tensor

class HindsightExperienceReplay: # HER 
    def __init__(self, her_cfg, goal_test_cfg):
        self.episode_transitions = []
        self._episode_transitions_info = []
        self.strategy = her_cfg['strategy'] if 'strategy' in her_cfg else 'future' 
        self.N = her_cfg['N']
        self.k = her_cfg['k']
        self.goal_test_cfg = goal_test_cfg
        
    def add_transition(self, transition:tuple, transition_info:dict):
        """ adding the real transition to the current episode's transitions list. 

        Args:
            transition (tuple): _description_
            info_for_reward_computation (dict): _description_
        """
        self.episode_transitions.append(transition)
        self._episode_transitions_info.append(transition_info)

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
                    new_goal_pose_gym = self._episode_transitions_info[sampled_transition_ts]['s_next_ee_pose_gym'] # this is the representation which is relevant for the reward computation
                    G[i] = new_goal_pose_gym # set ith "next state" as a goal in the additional goals list     
            
            return G
    
    def _make_modified_state_copy_with_new_goal(self, rlpt_agent, st_tensor, new_goal)-> np.ndarray:
        return rlpt_agent.make_modified_state(st_tensor, 'goal_pose',new_goal)  
        
    def _compute_rt_wrt_new_goal(self, rlpt_agent, step_duration, s_next_contact, s_next_pos_err_wrt_g_tag, s_next_rot_err_wrt_g_tag):
        return rlpt_agent.compute_reward(s_next_pos_err_wrt_g_tag, s_next_rot_err_wrt_g_tag,s_next_contact, step_duration)
                
    def optimize(self, rlpt_agent):
        
        T = len(self.episode_transitions)
        N = self.N if self.N != -1 else T  # num of optimization steps
        for t in range(T):
            G = self._sample_additional_goals(t, strategy=self.strategy, k=self.k) 
            st_tensor, at_idx_tensor, s_next_tensor = self.episode_transitions[t]
            transition_info = self._episode_transitions_info[t]
            
            for g_tag in G:
                                
                
                # compute new s(t) (with new goal pose "g-tag")
                g_tag_np_flatten = pose_as_ndarray(g_tag).flatten()  
                st_with_g_tag = self._make_modified_state_copy_with_new_goal(rlpt_agent, st_tensor, g_tag_np_flatten)
                
                # compute new s(t+1) (with new goal pose "g-tag")
                s_next_ee_pose_gym = transition_info['s_next_ee_pose_gym'] # pose where actually_reached
                s_next_ee_pos_error_wrt_new_goal = pos_error(s_next_ee_pose_gym.p, g_tag.p) # end effector position error (s(t+1))
                s_next_ee_rot_error_wrt_new_goal = rot_error(s_next_ee_pose_gym.r, g_tag.r)  # end effector rotation error (s(t+1))   
                s_next_is_terminal_wrt_new_goal = transition_info['s_next_contact_detected'] or goal_test(s_next_ee_pos_error_wrt_new_goal, s_next_ee_rot_error_wrt_new_goal, self.goal_test_cfg) 
                s_next_with_g_tag = None if s_next_is_terminal_wrt_new_goal else self._make_modified_state_copy_with_new_goal(rlpt_agent, s_next_tensor, g_tag_np_flatten)  
                
                # compute new r(t) (with new goal pose "g-tag")
                r_tag = self._compute_rt_wrt_new_goal(rlpt_agent, transition_info['step_duration'], transition_info['s_next_contact_detected'], s_next_ee_pos_error_wrt_new_goal, s_next_ee_rot_error_wrt_new_goal ) 
                
                # push r(t) to
                rlpt_agent.train_suit.memory.push(st_with_g_tag, at_idx_tensor, s_next_with_g_tag, as_1d_tensor([r_tag])) 
                
        for t in range(N):
            optim_meta_data = rlpt_agent.optimize() # TODO: Should make the C of fixed targets update support HER too 
            print('debug: optim meta data of HER updates')
            print(optim_meta_data)
        return optim_meta_data