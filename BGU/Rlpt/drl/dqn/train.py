"""
See: 
https://filebox.ece.vt.edu/~f15ece6504/slides/L26_RL.pdf
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
from dqn import DQN
from replay_memory import ReplayMemory, Transition




def set_seed(env, seed=1):
    random.seed(seed)
    torch.random.manual_seed(seed)
    # env.seed(seed)
    # env.action_space.seed(seed)
    # env.observation_space.seed(seed)


# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the  optimizer (aka alpha or step size)
# C is the target network update frequenty (set it to be the Q network's weights every C steps)
# T = Max step num in episode

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
# TAU = 0.005
LR = 1e-4
SEED = 42
C = 100 
N = 10000
T = 10000
set_seed(SEED)


# criterion = nn.SmoothL1Loss() # huber loss. Not in the original paper
criterion = nn.MSELoss()

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 1000
else:
    num_episodes = 50
    
# env = gym.make("CartPole-v1")
env: gym.Env = gym.make("CartPole-v1",render_mode="human")
# Get number of actions from gym action space
n_actions = env.action_space.n

# Initialize replay memory D to capacity N
memory = ReplayMemory(N, SEED)

# Get the number of state observations
state, info = env.reset()
n_observations = len(state)

# Initialize action-value function Q with random weights θ
policy_net:DQN = DQN(n_observations, n_actions).to(device) # Q

#Initialize target action-value function Q^ with weights θ- ← θ
target_net:DQN = DQN(n_observations, n_actions).to(device) # Q^
target_net.load_state_dict(policy_net.state_dict()) # θ- ← θ



steps_done = 0
episode_durations = []

def end_of_epiosode_steps():
    episode_durations.append(t + 1)
    plot_durations()

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
            return policy_net(state).max(1).indices.view(1, 1) # argmax a: Q∗(s_t, a; θ_t)
    
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
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())
            
def optimize_model():
    """
    https://daiwk.github.io/assets/dqn.pdf alg 1
    
    This is the optimization step of the model.
    The optimization step is the heart of the algorithm and its geniousity.
    We will use experience replay and optimize using 2 networks: Q network (updated) and targets network Q^ (older, fixed).

    """
    if len(memory) < BATCH_SIZE:
        return # optimization starts only if we accumulated at leaset BATCH_SIZE samples into the memory
    
    # A. SAMPLE a MINIBATCH (a random portion of the "train set")
    transitions = memory.sample(BATCH_SIZE) 
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    
    optimizer.zero_grad()
     
    # Pass batch previous states through the Q network. 
    # For each transition j in batch: (1<=j<=batch size): given pair s_j, a_j in batch, compute Q(s_j, a_j) using the Q network
    policy_net_q_batch: torch.Tensor = policy_net(state_batch).gather(1, action_batch) # https://pytorch.org/docs/stable/generated/torch.Tensor.gather.html#torch.Tensor.gather
    
    # Pass batch next states through the Q^ network.
    # For same batch, for each j compute the jth target (y_j) in batch using the  "targets network" (Q^) which has older fixed parameters θ−. 
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    target_net_next_states_q_for_all_actions = target_net(non_final_next_states)
    target_net_next_states_q_for_greedy_action = target_net_next_states_q_for_all_actions.max(1).values
    
    next_q_for_greedy_a_with_targets_net = torch.zeros(BATCH_SIZE, device=device) 
    with torch.no_grad():
        next_q_for_greedy_a_with_targets_net[non_final_mask] = target_net_next_states_q_for_greedy_action #  max a’Q^(s_j+1_, a’, θ−) (For each j, calculates the next state (s_j+1) Q^ (q of targtes) w.r to greedy action)
    # y = the targets vector = (y_1,...,y_batchsize)
    y = reward_batch +  GAMMA * next_q_for_greedy_a_with_targets_net   # y_j = r_j + γ * max a’Q^(s_j+1_, a’, θ−)
    y = y.unsqueeze(1)
    
    # loss calculation and greadient step
    loss = criterion(input=policy_net_q_batch, target=y)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1.0) # 1 is the maximal norm of each grad, like in paper
    optimizer.step()
    

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True) 
# optimizer = optim.SGD(policy_net.parameters(), lr=LR) # original paper
steps_done = 0
for i_episode in range(num_episodes): # https://daiwk.github.io/assets/dqn.pdf
    # Initialize the environment and get its state
    o, info = env.reset()
    s_t = torch.tensor(o, dtype=torch.float32, device=device).unsqueeze(0)
    
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
        
        optimize_model() # sample a minibatch from replay buffer and make an update step in the Q network

        # Every C steps update the Q network of targets to be as the frequently updating Q network of policy Q^ ← Q
        if steps_done % C == 0:
            target_net.load_state_dict(policy_net.state_dict())
                   
        if done:
            end_of_epiosode_steps()
            break
        
        s_t = s_next 

        
print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()