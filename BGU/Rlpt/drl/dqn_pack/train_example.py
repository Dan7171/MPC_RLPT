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
from dqn_pack.dqn import DQN
from dqn_pack.replay_memory import ReplayMemory, Transition
from art import text2art 
def set_seed(env, seed=1):
    random.seed(seed)
    torch.random.manual_seed(seed)
    # env.seed(seed)
    # env.action_space.seed(seed)
    # env.observation_space.seed(seed)

def select_action(state):
    """
    https://daiwk.github.io/assets/dqn.pdf alg 1
    Select an action from an epsilon greedy policy.
    With probability epsilon select a random action a_t otherwise select a_t = argmax a: Q∗(s_t, a; θ_t)
    """
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    select_greedily = sample > eps_threshold  
    
    if select_greedily: # best a (which maximizes current Q)
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return current(state).max(1).indices.view(1, 1) # argmax a: Q∗(s_t, a; θ_t)
    
    else: # random a (each action has a fair chance to be selected)
        return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long) # picking a random action

def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    
    is_ipython = 'inline' in matplotlib.get_backend()
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

def end_of_epiosode_steps():
    episode_durations.append(t + 1)
    plot_durations()
    
def optimize_model(ddqn=False):
    """
    https://daiwk.github.io/assets/dqn.pdf alg 1
    or with the modification to ddqn:  https://arxiv.org/pdf/1509.06461
    
    This is the optimization step of the model.
    The optimization step is the heart of the algorithm and its geniousity.
    We will use experience replay and optimize using 2 networks: Q network (updated) and targets network Q^ (older, fixed).

    """
    if len(memory) < BATCH_SIZE:
        return # optimization starts only if we accumulated at leaset BATCH_SIZE samples into the memory
    
    # sample minibatch
    transitions = memory.sample(BATCH_SIZE) 
    batch = Transition(*zip(*transitions))
    states, actions, rewards = torch.cat(batch.state), torch.cat(batch.action), torch.cat(batch.reward)
    
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool) # a vector in length = batch size. True where sj+1 is not final
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None]) # a vector in length <= batch size. Only the non final states sj+1 from batch.         
    
    # zero gradient
    optimizer.zero_grad()
    
    # Q(s, a, θ) (computing Q with updated (online) Q network, where a is the action which was taken in batch)
    q_values: torch.Tensor = current(states).gather(1, actions) # https://pytorch.org/docs/stable/generated/torch.Tensor.gather.html#torch.Tensor.gather
    with torch.no_grad():        
        if not ddqn: # standard dqn
            # Q(s',a',θ−) (computing Q with target(older)-network where s' is the next states observed in batch.)
            next_states_qs = target(non_final_next_states) 
            qs_for_target = next_states_qs.max(1).values # saving only the Qs of the maximizing actions

        else: # ddqn
            best_a = torch.argmax(current(non_final_next_states), dim=1, keepdim=False).unsqueeze(0).T # a' = argmax(Q(s', a')) from the Q-network. Action selected using Q (online network) 
            qs_for_target = target(non_final_next_states).gather(1, best_a) # Q_target values  using the best actions (using the offline "target" network)
            qs_for_target = qs_for_target.squeeze(1) # reshaping
        # Now take those Qs of non final states and set them in the batch vector including final and non final stats (we determine the final states Q values to 0)
        qs_for_target_with_final_states = torch.zeros(BATCH_SIZE, device=device)  # a vector in length = batch size
        # setting a value only at the non-final states. For the final states it reamins 0.
        qs_for_target_with_final_states[non_final_mask] = qs_for_target # If s' is not final: max a’Q^(s', a’, θ−). Else if its final: If final: set 0
            
    # y = the targets vector = (y_1,...,y_batchsize)
    y = rewards +  GAMMA * qs_for_target_with_final_states # targtes. expected q values given. our "labeled" data     
    
    # loss calculation and greadient step
    loss = criterion(input=q_values, target=y.unsqueeze(1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(current.parameters(), 1.0) # 1 is the maximal norm of each grad, like in paper
    optimizer.step()




# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the  optimizer (aka alpha or step size)
# C is the target network update frequenty (set it to be the Q network's weights every C steps)
# T = Max step num in episode

# user argumaents:
# >>>>>
ddqn = True # changing the y values compuation to convert the algorithm from dqn to ddqn. See https://arxiv.org/pdf/1509.06461
# <<<<<<
print(text2art(f"{'DDQN' if ddqn else 'DQN'}"))
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
LR = 1e-4
SEED = 42
C = 100 
N = 10000
T = 10000
set_seed(SEED)

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.ion()

device = torch.device("cuda" if torch.cuda.is_available() else  "mps" if torch.backends.mps.is_available() else "cpu")
if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 1000
else:
    num_episodes = 50

env: gym.Env = gym.make("CartPole-v1",render_mode="human") # env = gym.make("CartPole-v1")
n_actions = env.action_space.n # Get number of actions from gym action space
state, info = env.reset()
n_observations = len(state)
episode_durations = []

####### DQN: https://daiwk.github.io/assets/dqn.pdf ##########

# criterion = nn.SmoothL1Loss() # huber loss. Not in the original paper
criterion = nn.MSELoss()
memory = ReplayMemory(N, SEED) # Initialize replay memory D to capacity N
steps_done = 0 # in total, over all episodes    
# Initialize action-value function Q with random weights θ
current:DQN = DQN(n_observations, n_actions).to(device) # Q
#Initialize target action-value function Q^ with weights θ- ← θ
target:DQN = DQN(n_observations, n_actions).to(device) # Q^
target.load_state_dict(current.state_dict()) # θ- ← θ
optimizer = optim.AdamW(current.parameters(), lr=LR, amsgrad=True) 
# optimizer = optim.SGD(current.parameters(), lr=LR) # original paper


for i_episode in range(num_episodes): # https://daiwk.github.io/assets/dqn.pdf
    # Initialize the environment and get its state
    o, info = env.reset()
    s_t = torch.tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
    # print(f'episde: {i_episode}')
    # for t in count(): # iterating time steps of current episode
    for t in range(1, T+1):  
        steps_done += 1
        # select action a_t according to €-greedy policy (With probability € select a random action at)
        a_t = select_action(s_t)  
        # execute action a_t in environment
        o_next, r_t, terminated, truncated, _ = env.step(a_t.item()) # o_t+1, r_t, mdp's terminal state reached, forced to stop (outside of mdp's scope)   
        r_t = torch.tensor([r_t], device=device)
        done = terminated or truncated 
        s_next = None
        if not terminated:
            s_next = torch.tensor(o_next, dtype=torch.float32, device=device).unsqueeze(0) # s_t+1 ← o_t+1 (fully observable)
        
        # store transition (st, at, st+1, rt) in replay memory D. To make data to train the Q network 
        memory.push(s_t, a_t, s_next, r_t)   
        
        optimize_model(ddqn) # sample a random minibatch of transitions (s_j, a_j, s_j+1, r_j) from D, compute errors and make a gradient step.   
        if steps_done % C == 0: # Every C steps update the Q network of targets to be as the frequently updating Q network of policy Q^ ← Q
            target.load_state_dict(current.state_dict())
                   
        if done:
            end_of_epiosode_steps()
            break
        
        s_t = s_next 

        
print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()