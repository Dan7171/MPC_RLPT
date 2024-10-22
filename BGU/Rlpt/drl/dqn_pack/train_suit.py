"""
Deep Q learning with experience replay, seperate target network, and gradient clipping.
Train DDQN if using flag ddqn=True else trains DQN.

See: 
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
https://filebox.ece.vt.edu/~f15ece6504/slides/L26_RL.pdf
https://youtu.be/UoPei5o4fps?si=BQdBcYCl60NGhGtJ
https://arxiv.org/pdf/1312.5602

"""
from graphviz import render
import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import torch.nn as nn
from itertools import count
from BGU.Rlpt.drl.dqn_pack.replay_memory import ReplayMemory, Transition
from BGU.Rlpt.drl.dqn_pack.dqn import DQN

from art import text2art 


class trainSuit:
    """
    
    everything which is needed for training DQN/DDQN on any mdp
    
    """
    
    def __init__(self, state_dim_flatten, n_actions,  ddqn=True, seed=42, batch_size=64,gamma=0.99, eps_start = 0.9, eps_end=0.05, eps_decay=1000,learning_rate=1e-4,
                C=100,N=10000, T=10000, criterion=nn.MSELoss, optimizer=optim.AdamW):
        """Initializing a dqn/ddqn network 

        Args:
            state_dim_flatten (_type_): _description_
            n_actions (_type_): _description_
            ddqn (bool, optional): _description_. Defaults to True.
            seed (int, optional): _description_. Defaults to 42.
            batch_size (int, optional): _description_. Defaults to 128.
            gamma (float, optional): _description_. Defaults to 0.99.
            eps_start (float, optional): _description_. Defaults to 0.9.
            eps_end (float, optional): _description_. Defaults to 0.05.
            eps_decay (int, optional): _description_. Defaults to 1000.
            learning_rate (_type_, optional): _description_. Defaults to 1e-4.
            C (int, optional): _description_. Defaults to 100.
            N (int, optional): _description_. Defaults to 10000.
            T (int, optional): _description_. Defaults to 10000.
            criterion (_type_, optional): _description_. Defaults to nn.MSELoss.
            optimizer (_type_, optional): _description_. Defaults to optim.AdamW.
        """
        # BATCH_SIZE is the number of transitions sampled from the replay buffer
        # GAMMA is the discount factor as mentioned in the previous section
        # EPS_START is the starting value of epsilon
        # EPS_END is the final value of epsilon
        # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
        # TAU is the update rate of the target network
        # learning_rate is the learning rate of the  optimizer (aka alpha or step size)
        # C is the target network update frequenty (set it to be the Q network's weights every C steps)
        # T = Max step num in episode
        
        self.ddqn = ddqn
        self.seed = seed
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.learning_rate = learning_rate
        self.C = C
        self.N = N
        self.T = T
        self.n_actions = n_actions
        self.action_indices = range(n_actions) # these are the action ids. Corresponding to Q networks output layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else  "mps" if torch.backends.mps.is_available() else "cpu")

        
        # criterion = nn.SmoothL1Loss() # huber loss. Not in the original paper
        self.criterion = criterion()
        self.memory = ReplayMemory(self.N, self.seed) # Initialize replay memory D to capacity N
        self.steps_done = 0 # in total, over all episodes    
        
        # Initialize first Q network with random weights θ ("current"/ "online" network)
        self.current:DQN = DQN(state_dim_flatten, self.n_actions).to(self.device) # Q(θ)
        
        # Initialize a second Q network with weights θ- ("target network"/"Q^"/"offline network"). 
        self.target:DQN = DQN(state_dim_flatten, self.n_actions).to(self.device) # Q(θ-) aka θ^
        self.target.load_state_dict(self.current.state_dict()) #setting them initally to θ (θ- ← θ)
        
        if optimizer == optim.AdamW:
            self.optimizer = optimizer(self.current.parameters(), lr=self.learning_rate, amsgrad=True) 
        
        elif optimizer == optim.SGD:
            self.optimizer = optimizer(self.current.parameters(), lr=self.learning_rate) # original paper



    def set_seed(self, env, seed=1):
        random.seed(seed)
        torch.random.manual_seed(seed)

    def select_action(self, state):
        """
        https://daiwk.github.io/assets/dqn.pdf alg 1
        Select an action from an epsilon greedy policy.
        With probability epsilon select a random action a_t otherwise select a_t = argmax a: Q∗(s_t, a; θ_t)
        
        Return the idx of the aciton and the action
        """
        
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        select_greedily = sample > eps_threshold  
        
        if select_greedily: # best a (which maximizes current Q)
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                # action_idx_tensor = self.current(state).max().indices.view(1, 1) # argmax a: Q∗(s_t, a; θ_t)
                action_idx_tensor = torch.argmax(self.current(state)) # [index]
                action_idx = action_idx_tensor.item() # index
                
        else: # random a (each action has a fair chance to be selected)
            action_idx = random.randint(0, self.n_actions -1)
            action_idx_tensor = torch.tensor([[action_idx]], device=self.device, dtype=torch.long)
        
        # action = self.action_space[action_idx]
        # return action_idx_tensor, action # picking a random action
        
        return action_idx_tensor # that index is the id of the action 
    
    def optimize(self):
        """
        Sample a random minibatch of transitions from the shape (s, a, s', r) from D, 
        compute error in Q^ w.r to target Q^s as the "true values",  and make a gradient step.
        
        sources:    
        https://daiwk.github.io/assets/dqn.pdf alg 1
        or with the modification to ddqn:  https://arxiv.org/pdf/1509.06461
        This is the optimization step of the model.
        We will use experience replay and optimize using 2 networks: Q network (updated) and targets network Q^ (older, fixed).

        """
        if len(self.memory) < self.batch_size:
            return # optimization starts only if we accumulated at leaset self.batch_size samples into the memory
        
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
        
        # loss calculation and greadient step
        loss = self.criterion(input=q_values, target=y.unsqueeze(1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.current.parameters(), 1.0) # 1 is the maximal norm of each grad, like in paper
        self.optimizer.step()
        
        self.steps_done += 1
        if self.steps_done % self.C == 0: # Every C steps update the Q network of targets to be as the frequently updating Q network of policy Q^ ← Q
            self.target.load_state_dict(self.current.state_dict())
                
        return loss

