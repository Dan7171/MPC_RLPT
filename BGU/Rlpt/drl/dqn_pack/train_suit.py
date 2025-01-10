"""
Deep Q learning with experience replay, seperate target network, and gradient clipping.
Train DDQN if using flag ddqn=True else trains DQN.

See: 
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
https://filebox.ece.vt.edu/~f15ece6504/slides/L26_RL.pdf
https://youtu.be/UoPei5o4fps?si=BQdBcYCl60NGhGtJ
https://arxiv.org/pdf/1312.5602

"""
from curses import color_pair
from typing import Union
from graphviz import render
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from itertools import count
from BGU.Rlpt.DebugTools.globs import GLobalVars
from BGU.Rlpt.drl.dqn_pack.replay_memory import ReplayMemory, Transition
from BGU.Rlpt.drl.dqn_pack.dqn import DQN
from BGU.Rlpt.configs.default_main import load_config_with_defaults

from art import text2art

from BGU.Rlpt.utils.utils import color_print 


class trainSuit:
    """
    
    everything which is needed for training DQN/DDQN on any mdp
    
    """
    
    
    def __init__(self, state_dim_flatten, n_actions, episode_idx, max_episode,  ddqn=True, seed=42, batch_size=256,gamma=0.99,
                eps_start = 0.999, eps_end=0.05, eps_decay=100000,learning_rate=0.01,
                C=100,N=100000, T=10000, criterion=nn.MSELoss, optimizer=optim.AdamW):
        """Initializing a dqn/ddqn network 

        Args:
            state_dim_flatten (_type_): _description_
            n_actions (_type_): _description_
            ddqn (bool, optional): _description_. Defaults to True.
            seed (int, optional): _description_. Defaults to 42.
            batch_size (int, optional): _description_. Defaults to 128.
            gamma (float, optional): _description_. Defaults to 0.99.
            # eps_start (float, optional): _description_. Defaults to 0.9.
            # eps_end (float, optional): _description_. Defaults to 0.05.
            # eps_decay (int, optional): _description_. Defaults to 1000.
            learning_rate (_type_, optional): _description_. Defaults to 1e-4.
            C (int, optional): _description_. Defaults to 100.
            N (int, optional): _description_. Defaults to 10000.
            T (int, optional): _description_. Defaults to 10000.
            criterion (_type_, optional): _description_. Defaults to nn.MSELoss.
            optimizer (_type_, optional): _description_. Defaults to optim.AdamW.
        """
        # BATCH_SIZE is the number of transitions sampled from the replay buffer
        # GAMMA is the discount factor as mentioned in the previous section
        ## EPS_START is the starting value of epsilon
        ## EPS_END is the final value of epsilon
        ## EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
        # TAU is the update rate of the target network
        # learning_rate is the learning rate of the  optimizer (aka alpha or step size)
        # C is the target network update frequenty (set it to be the Q network's weights every C steps)
        # T = Max step num in episode
        cfg = GLobalVars.rlpt_cfg['agent']['train_suit']
        self.ddqn = ddqn
        self.seed = seed
        self.gamma = cfg['gamma'] if 'gamma' in cfg else gamma
        self.batch_size = cfg['batch_size'] if 'batch_size' in cfg else batch_size
        # self.eps_start = eps_start
        self.eps_end = eps_end
        # self.eps_decay = cfg['eps_decay'] if 'eps_decay' in cfg else eps_decay
        
        self.eps_decay = cfg['eps_decay']
        if not self.eps_decay:
            self.current_eps = cfg['default_eps']
        self.episode_idx = episode_idx
        self.max_episode = max_episode
        self.learning_rate = learning_rate
        self.C = C
        self.N = cfg['N'] if N in cfg else N
        self.T = T
        self.n_actions = n_actions
        self.action_indices = range(n_actions) # these are the action ids. Corresponding to Q networks output layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else  "mps" if torch.backends.mps.is_available() else "cpu")
        self.clipping = cfg['gradient_clipping'] if 'gradient_clipping' in cfg else False
        
        # criterion = nn.SmoothL1Loss() # huber loss. Not in the original paper
        self.criterion = criterion()
        self.memory = ReplayMemory(self.N, self.seed) # Initialize replay memory D to capacity N
        # self.steps_done = 0 # in total, over all episodes    
        
        # Initialize first Q network with random weights θ ("current"/ "online" network)
        self.current:DQN = DQN(state_dim_flatten, self.n_actions).to(self.device) # Q(θ)
        
        # Initialize a second Q network with weights θ- ("target network"/"Q^"/"offline network"). 
        self.target:DQN = DQN(state_dim_flatten, self.n_actions).to(self.device) # Q(θ-) aka θ^
        self.target.load_state_dict(self.current.state_dict()) #setting them initally to θ (θ- ← θ)
        
        if optimizer == optim.AdamW:
            self.optimizer = optimizer(self.current.parameters(), lr=self.learning_rate, amsgrad=True) 
        
        elif optimizer == optim.SGD:
            self.optimizer = optimizer(self.current.parameters(), lr=self.learning_rate) # original paper


    # def set_episode_idx(self,  episode_idx):
    #     self.episode_idx = episode_idx
    
    def set_seed(self, env, seed=1):
        random.seed(seed)
        torch.random.manual_seed(seed)

    def compute_grad_norm(self):
        total_grad_norm = 0.0
        # Iterate through model parameters to compute gradient norms
        for param in self.current.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)  # L2 norm of the gradient
                total_grad_norm += param_norm.item() ** 2

        # Final gradient norm across all parameters
        total_grad_norm = total_grad_norm ** 0.5
        return total_grad_norm

    def pick_greedy_no_grads(self, state):
        with torch.no_grad():
            Q_all_actions = self.current(state)
            Q_all_actions_with_idx = [(Q_all_actions[i].item(), i) for i in range(len(Q_all_actions))] # all actions q value with action idx
            max_allowed_q, max_allowed_q_idx = max(Q_all_actions_with_idx, key=lambda t: t[0]) # selecting the maximizing tuple based on the q value 
            return max_allowed_q, max_allowed_q_idx
        
    def select_action_idx(self, state:torch.Tensor, indices_to_filter_out: set=set(), training:bool=True) -> Union[torch.Tensor, dict]:
        """
        https://daiwk.github.io/assets/dqn.pdf alg 1
        Select an action from an epsilon greedy policy.
        With probability epsilon select a random action a_t otherwise select a_t = argmax a: Q∗(s_t, a; θ_t)
        
        Return the idx of the aciton and the action
        """
        
        all_action_indices:set = set(range(self.n_actions))
        allowed_actions_indices = all_action_indices - indices_to_filter_out
        sample = random.random()
        if self.eps_decay: 
            self.current_eps = max(self.eps_end, (self.max_episode - self.episode_idx) / self.max_episode) 
        if training:
            greedy_choice = sample > self.current_eps
        else: # deployment mode, taking the best action
            greedy_choice = True
        with torch.no_grad():
            Q_all_actions = self.current(state)
            if greedy_choice: # best a (which maximizes current Q)
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                # action_idx_tensor = self.current(state).max().indices.view(1, 1) # argmax a: Q∗(s_t, a; θ_t)
                Q_all_actions_with_idx = [(Q_all_actions[i].item(), i) for i in range(len(Q_all_actions))] # all actions q value with action idx
                allowed_Q_values_with_idx = [qi for qi in Q_all_actions_with_idx if qi[1] in allowed_actions_indices] # filter out the not allowed actions 
                max_allowed_q, max_allowed_q_idx = max(allowed_Q_values_with_idx, key=lambda t: t[0]) # selecting the maximizing tuple based on the q value 
                # action_idx_tensor = torch.argmax(Q_all_actions) # [index]
                # action_idx = action_idx_tensor.item() # index
                action_idx = max_allowed_q_idx
                picked_q = max_allowed_q
                
                # print(f'best action q: {torch.max(Q_all_actions)}')
                # print(f'arg max q(s,a): {torch.max(Q_all_actions)}, max allowed q(s,a): {max_allowed_q}')
                
            else: # random action (each action has a fair chance to be selected)
                # action_idx = random.randint(0, self.n_actions -1)
                action_idx = random.choice(list(allowed_actions_indices)) 
                picked_q = Q_all_actions[action_idx]
            # color_print(f'Q(s,a) = {picked_q}, a = {action_idx}')
                
        # print(f'max allowed q(s,a): {picked_q:{.3}f} (max q(s,a): {torch.max(Q_all_actions):{.3}f})')
        # print(f'max q(s,a): {torch.max(Q_all_actions):{.3}f})')
        
        action_idx_tensor = torch.tensor([[action_idx]], device=self.device, dtype=torch.long)
        meta_data = {}
        meta_data['q']= picked_q.item() if type(picked_q) == torch.Tensor else picked_q  
        if training:
            meta_data['eps']= self.current_eps
            meta_data['is_random'] = not greedy_choice 
        
        return action_idx_tensor, meta_data # that index is the id of the action 
    
    def optimize(self, episode_ts, max_norm=1.0):
        """
        Sample a random minibatch of transitions from the shape (s, a, s', r) from D, 
        compute error in Q^ w.r to target Q^s as the "true values",  and make a gradient step.
        # max norm: 1 is the maximal norm of each grad, like in paper
        sources:    
        https://daiwk.github.io/assets/dqn.pdf alg 1
        or with the modification to ddqn:  https://arxiv.org/pdf/1509.06461
        This is the optimization step of the model.
        We will use experience replay and optimize using 2 networks: Q network (updated) and targets network Q^ (older, fixed).

        """
        
        
        meta_data = {'raw_grad_norm': 0, 'clipped_grad_norm':0.0, 'use_clipped':self.clipping, 'loss' :0}

        if len(self.memory) < self.batch_size:
            return meta_data # optimization starts only if we accumulated at leaset self.batch_size samples into the memory
        
        # sample minibatch
        transitions = self.memory.sample(self.batch_size) 
        batch = Transition(*zip(*transitions))
        states, actions, rewards = torch.cat(batch.state), torch.cat(batch.action), torch.cat(batch.reward)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool) # a vector in length = batch size. True where sj+1 is not final
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]) # a vector in length <= batch size. Only the non final states sj+1 from batch.         
        
        # zero gradient
        self.optimizer.zero_grad()
        
        # Q(s, a, θ) (computing Q with updated (online, current) Q network, where a is the action which was taken in batch)
        q_values: torch.Tensor = self.current(states).gather(1, actions) # https://pytorch.org/docs/stable/generated/torch.Tensor.gather.html#torch.Tensor.gather
        with torch.no_grad():        
            if not self.ddqn: # standard dqn
                # Q(s',a',θ−) (computing Q with target(older)-network where s' is the next states observed in batch.)
                next_states_qs = self.target(non_final_next_states) 
                qs_for_target = next_states_qs.max(1).values # saving only the Qs of the maximizing actions

            else: # ddqn
                best_a = torch.argmax(self.current(non_final_next_states), dim=1, keepdim=False).unsqueeze(0).T # a' = argmax(Q(s', a')) from the Q-network. Action selected using Q (online network) 
                qs_for_target = self.target(non_final_next_states).gather(1, best_a) # Q_target values  using the best actions (using the offline "target" network)
                qs_for_target = qs_for_target.squeeze(1) # reshaping
            # Now take those Qs of non final states and set them in the batch vector including final and non final stats (we determine the final states Q values to 0)
            qs_for_target_with_final_states = torch.zeros(self.batch_size, device=self.device)  # a vector in length = batch size
            # setting a value only at the non-final states. For the final states it reamins 0.
            qs_for_target_with_final_states[non_final_mask] = qs_for_target # If s' is not final: max a’Q^(s', a’, θ−). Else if its final: If final: set 0
                
        # y = the targets vector = (y_1,...,y_batchsize)
        y = rewards +  self.gamma * qs_for_target_with_final_states # targtes. expected q values given. our "labeled" data     
        
        # compute loss
        loss = self.criterion(input=q_values, target=y.unsqueeze(1)) # update current (Q network) weights in such way that we minimize the prediction error, w.r. to Q targets as our "y_real" with Q network predictiosn as "y_pred"
        # compute grads
        loss.backward() 
        meta_data['raw_grad_norm'] = self.compute_grad_norm()
        
        if self.clipping:
            torch.nn.utils.clip_grad_norm_(self.current.parameters(), max_norm) 
            meta_data['clipped_grad_norm'] = self.compute_grad_norm() # clipped gradient norm

        # gradient step
        self.optimizer.step()
        
        
        if episode_ts % self.C == 0: # Every C steps update the Q network of targets to be as the frequently updating Q network of policy Q^ ← Q
            # color_print(f'debug episodes time step= {episode_ts}')
            self.target.load_state_dict(self.current.state_dict())
        
        
        meta_data['loss'] = loss.item() 
        
        return meta_data

