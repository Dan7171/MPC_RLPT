
import matplotlib.pyplot as plt
from collections import deque
from typing import Any, Deque, Dict, List, Tuple, Union
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from IPython.display import clear_output
from IPython.display import HTML, display
from torch.nn.utils import clip_grad_norm_
from BGU.Rlpt.drl.rainbow_rlpt.network import Network
from BGU.Rlpt.drl.rainbow_rlpt.prioritized_replay_buffer import PrioritizedReplayBuffer
from BGU.Rlpt.drl.rainbow_rlpt.replay_buffer import ReplayBuffer
from rlpt_agent_base import rlptAgentBase

class DQNAgent(rlptAgentBase):
    """DQN Agent interacting with environment.
    
    Attribute:
        memory (PrioritizedReplayBuffer): replay memory to store transitions
        batch_size (int): batch size for sampling
        target_update (int): period for target model's hard update
        gamma (float): discount factor
        dqn (Network): model to train and select actions
        dqn_target (Network): target model to update
        optimizer (torch.optim): optimizer for training dqn
        transition (list): transition information including 
                           state, action, reward, next_state, done
        v_min (float): min value of support
        v_max (float): max value of support
        atom_size (int): the unit number of support
        support (torch.Tensor): support for categorical dqn
        use_n_step (bool): whether to use n_step memory
        n_step (int): step number to calculate n-step td error
        memory_n (ReplayBuffer): n-step replay buffer
    """

    def __init__(
        self, 

        super_params: dict, # new 
        memory_size: int,
        batch_size: int,
        target_update: int,
        gamma: float = 0.99,
        # PER parameters
        alpha: float = 0.2,
        beta: float = 0.6,
        prior_eps: Union[float,str] = 1e-6,
        # Categorical DQN parameters
        v_min: float = 0.0,
        v_max: float = 200.0,
        atom_size: int = 51,
        # N-step Learning
        n_step: int = 3,
        device_dtype: str = "torch.float32" # new

    ):
        """Initialization.
        
        Args:
            super_params': dict
            memory_size (int): length of memory
            batch_size (int): batch size for sampling
            target_update (int): period for target model's hard update
            lr (float): learning rate
            gamma (float): discount factor
            alpha (float): determines how much prioritization is used
            beta (float): determines how much importance sampling is used
            prior_eps (float): guarantees every transition can be sampled
            v_min (float): min value of support
            v_max (float): max value of support
            atom_size (int): the unit number of support
            n_step (int): step number to calculate n-step td error
            
        """
        super(DQNAgent, self).__init__(**super_params)  # initialize super
        
        # obs_dim = env.observation_space.shape[0]
        obs_dim:int = self.calc_obs_dim() # from super
        # action_dim = env.action_space.n
        action_dim:int = self.calc_action_dim() # from super
        # self.env = env
        self.batch_size = batch_size
        self.target_update = target_update
        self.gamma = gamma
        # NoisyNet: All attributes related to epsilon are removed
        
        # device: cpu / gpu
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        dtype_dict = {'torch.float32':torch.float32,'torch.float64':torch.float64}
        self.device_dtype = dtype_dict[device_dtype]
        
        # PER
        # memory for 1-step Learning
        self.beta = beta
        self.prior_eps = float(prior_eps) # in case its a string (like "1e-6")
        self.memory = PrioritizedReplayBuffer(
            obs_dim, memory_size, batch_size, alpha=alpha, gamma=gamma
        )
        
        
        # memory for N-step Learning
        self.use_n_step = True if n_step > 1 else False
        if self.use_n_step:
            self.n_step = n_step
            self.memory_n = ReplayBuffer(
                obs_dim, memory_size, batch_size, n_step=n_step, gamma=gamma
            )
            
        # Categorical DQN parameters
        self.v_min = v_min
        self.v_max = v_max
        self.atom_size = atom_size
        self.support = torch.linspace(
            self.v_min, self.v_max, self.atom_size
        ).to(self.device, dtype=self.device_dtype)

        # networks: dqn, dqn_target
        self.dqn = Network(
            obs_dim, action_dim, self.atom_size, self.support
        ).to(self.device,dtype=self.device_dtype)
        self.dqn_target = Network(
            obs_dim, action_dim, self.atom_size, self.support
        ).to(self.device,dtype=self.device_dtype)
        self.dqn_target.load_state_dict(self.dqn.state_dict())
        self.dqn_target.eval()
        
        # optimizer
        self.optimizer = optim.Adam(self.dqn.parameters())

        # transition to store in memory
        self.transition = list()
        
        # mode: train / test
        # self.is_test = False
        self.is_test = not self.training_mode # from super
        
        
    def _select_action(self,st:torch.Tensor, *args, **kwargs)-> Tuple[int,dict]:
        """Select an action from the input state."""
        # NoisyNet: no epsilon greedy action selection
        at_meta_data = {} # due to the usage in noisy net
        
        q_wt_st_all = self.dqn(
            # torch.FloatTensor(st).to(self.device)
            st.to(self.device,dtype=self.device_dtype)
        ) # all q values for s(t) and w(t)
        at_meta_data['q(w,st,all)'] = list(q_wt_st_all[0].cpu().detach().numpy())
        selected_action_idx = q_wt_st_all.argmax() # index of maximizing action
        selected_action_idx = selected_action_idx.detach().cpu().numpy() 
        
        if not self.is_test:
            self.transition = [st, selected_action_idx]
        
        # return selected_action index and its meta data
        return int(selected_action_idx), at_meta_data # turns an np array of 1 item to its val (an integer)
    

    # def step(self, action: np.ndarray) -> Tuple[np.ndarray, np.float64, bool]:
    #     """Take an action and return the response of the env."""
    #     next_state, reward, terminated, truncated, _ = self.env.step(action)
    #     done = terminated or truncated    
    #     self.store_transition(next_state, reward, done)
    #     return next_state, reward, done
    
    def store_transition(self, next_state, reward, terminated):
        """Was seperated from step so it will be easy to use it from mpc rlpt"""
        
        
        if not self.is_test:
            self.transition += [reward, next_state, terminated]
            # N-step transition
            if self.use_n_step:
                one_step_transition = self.memory_n.store(*self.transition)
            # 1-step transition
            else:
                one_step_transition = self.transition

            # add a single step transition
            if one_step_transition:
                self.memory.store(*one_step_transition)
    

    def update_model(self) -> torch.Tensor:
        """Update the model by gradient descent."""
        # PER needs beta to calculate weights
        samples = self.memory.sample_batch(self.beta) # minibatch of transitions, sampled with importance sampling
        weights = torch.FloatTensor( # weight of each transition in batch (wi in paper)
            samples["weights"].reshape(-1, 1)
        ).to(self.device,dtype=self.device_dtype)
        indices = samples["indices"] 
        
        
        # 1-step Learning loss
        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)  
        
        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights) 
        
        # N-step Learning loss
        # we are gonna combine 1-step loss and n-step loss so as to
        # prevent high-variance. The original rainbow employs n-step loss only.
        if self.use_n_step:
            gamma = self.gamma ** self.n_step
            samples = self.memory_n.sample_batch_from_idxs(indices)
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
            elementwise_loss += elementwise_loss_n_loss
            
            # PER: importance sampling before average
            loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.dqn.parameters(), 10.0)
        self.optimizer.step()
        
        
        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy() # computes loss for priority pi (td-error i)
        new_priorities = loss_for_prior + self.prior_eps # priority pi = td_err(i) + epsilon
        self.memory.update_priorities(indices, new_priorities) # pi -> pi ** alpha (see paper)
        
        # NoisyNet: reset noise
        self.dqn.reset_noise()
        self.dqn_target.reset_noise()

        return loss.item()
    
    def _update_beta(self, training_steps_done, traning_steps_limit):
        fraction = min(training_steps_done / traning_steps_limit, 1.0)
        self.beta = self.beta + fraction * (1.0 - self.beta)
 
    def _training_step_post_ops(self, *args, **kwargs) -> Any:
        
        meta_data = {'optimization': {'loss': -1}} 
        training_steps_done = self.get_training_steps_done()
        self._update_beta(training_steps_done, (kwargs['max_episode_index'] + 1) * kwargs['max_ts_per_episode'])
        if len(self.memory) >= self.batch_size:
            loss = self.update_model()
            meta_data['optimization']['loss'] = loss
            if training_steps_done % self.target_update == 0:
                self._target_hard_update()
                
        return meta_data
    
    def _training_episode_post_ops(self,*args, **kwargs) -> Any:
        pass
    
    def _test_episode_post_ops(self,*args, **kwargs) -> Any:
        pass
    
   
    
        
    
    
        
        
        
    # def train(self, num_frames: int, plotting_interval: int = 200):
    #     """Train the agent."""
    #     self.is_test = False
        
    #     state, _ = self.env.reset(seed=self.seed)
    #     update_cnt = 0
    #     losses = []
    #     scores = []
    #     score = 0
            
    #     for frame_idx in range(1, num_frames + 1):
    #         if frame_idx%100 == 0:
    #             print(f'debug: reached {frame_idx} traning steps')
    #         action = self.select_action(state)
    #         next_state, reward, done = self.step(action)

    #         state = next_state
    #         score += reward
            
    #         # NoisyNet: removed decrease of epsilon
            
    #         # PER: increase beta
    #         fraction = min(frame_idx / num_frames, 1.0)
    #         self.beta = self.beta + fraction * (1.0 - self.beta)

    #         # if episode ends
    #         if done:
    #             state, _ = self.env.reset(seed=self.seed)
    #             scores.append(score)
    #             score = 0

    #         # if training is ready
    #         if len(self.memory) >= self.batch_size:
    #             loss = self.update_model()
    #             losses.append(loss)
    #             update_cnt += 1
                
    #             # if hard update is needed
    #             if update_cnt % self.target_update == 0:
    #                 self._target_hard_update()

    #         # plotting
    #         if frame_idx % plotting_interval == 0:
    #             # self._plot(frame_idx, scores, losses)
    #             self.plot(frame_idx, scores, losses)
                
    #     self.env.close()
                
    # def test(self, video_folder: str) -> None:
    #     """Test the agent."""
    #     self.is_test = True
        
    #     # for recording a video
    #     naive_env = self.env
    #     # self.env = gym.wrappers.RecordVideo(self.env, video_folder=video_folder)
    #     self.env = env = gym.make("CartPole-v1", max_episode_steps=200, render_mode="human")

    #     state, _ = self.env.reset(seed=self.seed)
    #     done = False
    #     score = 0
        
    #     while not done:
    #         action = self.select_action(state)
    #         next_state, reward, done = self.step(action)

    #         state = next_state
    #         score += reward
        
    #     print("score: ", score)
    #     self.env.close()
        
    #     # reset
    #     self.env = naive_env

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        """Return categorical dqn loss."""
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device,dtype=self.device_dtype)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device,dtype=self.device_dtype)
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device,dtype=self.device_dtype)
        done = torch.FloatTensor(samples["done"].reshape(-1, 1)).to(device,dtype=self.device_dtype)
        
        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            next_action = self.dqn(next_state).argmax(1)
            next_dist = self.dqn_target.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward + (1 - done) * gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                .unsqueeze(1)
                .expand(self.batch_size, self.atom_size)
                .to(self.device)
                # .to(self.device, dtype=self.device_dtype)
            )
            # proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist = torch.zeros(next_dist.size(), device=self.device,dtype=self.device_dtype)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.dqn.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss

    def _target_hard_update(self):
        """Hard update: target <- local."""
        self.dqn_target.load_state_dict(self.dqn.state_dict())
                
    # def _plot(
    def plot(
        self, 
        training_steps_done: int,
        rewards: List[float], 
        losses: List[float],
        make_figure
    ):
        """Plot the training progresses."""
        
        clear_output(True)
        if make_figure:
            plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('time step %s. score: %s' % (training_steps_done, np.mean(rewards[-10:])))
        
        plt.plot(rewards)
        plt.subplot(132)
        plt.title('loss')
        plt.plot(losses)
        # plt.show()
        plt.draw()   
        plt.pause(0.05)

        
        
    def _load(self, checkpoint:dict) -> None: 
        """ inheritance implementation

        Args:
            checkpoint (dict): _description_
        """
        self.dqn.load_state_dict(checkpoint['dqn']) 
        if not self.is_test:
            self.dqn_target.load_state_dict(checkpoint['dqn_target'])
            self.memory = checkpoint['memory']
            
            

    def _get_items_to_save(self,*args, **kwargs):
        """ inheritance implementation

        Args:
            checkpoint (dict): _description_
        """
        
        if not self.is_test:
            items_to_save = {
                'dqn': self.dqn.state_dict(),
                'dqn_target': self.dqn_target.state_dict(),
                'memory': self.memory
            }
        else:
            items_to_save = {}
        return items_to_save
    
    
    