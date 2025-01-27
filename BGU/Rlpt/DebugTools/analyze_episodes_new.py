from cProfile import label
import copy
import matplotlib
# matplotlib.use('Agg')  # Use a non-interactive backend

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tomlkit import item
# path = '/home/dan/MPC_RLPT/BGU/Rlpt/favorite_models/2025:01:25(Sat)18:56:12/training_etl.csv'
path = '/home/dan/MPC_RLPT/BGU/Rlpt/favorite_models/2025:01:25(Sat)23:05:28/training_etl.csv'
df = pd.read_csv(path)
print(df.head)
print(df.nunique())

def _parse_qvals(qvals_str):
    qvals_list = qvals_str[1:-1].split(', ')
    qvals_list = [float(x) for x in qvals_list]
    return qvals_list

def parse_qvals_col(df):
    qvals_col = df['atMDq(w,st,all)']
    qvals_col = qvals_col.apply(lambda x: _parse_qvals(x))
    return qvals_col
    
def rt_plot_grouped(df):
    episodes = df.groupby('ep_id')
    plt.figure()
    for i, ng_tuple in enumerate(episodes):
        name, group = ng_tuple 
        plt.plot(group['rt'])
    plt.title('r(t) (color means episode)')
def rt_bar(df):
    plt.bar(df['rt'].index,df['rt'])

def get_qvals_at_idx(df,idx):
    qs = _parse_qvals(df['atMDq(w,st,all)'][df.index[idx]])
    return qs

def q0_star_plot(df):
    episodes = df.groupby('ep_id')
    plt.figure()
    q0_star = []
    for i, ng_tuple in enumerate(episodes):
        name, group = ng_tuple 
        s0_all_q = get_qvals_at_idx(group,0)
        q0_star.append(np.argmax(s0_all_q))
    plt.title('q*(s0)')
    plt.plot(q0_star)

def plt_with_label_markers(y,label_markers,x=None):
    if x is None:
        x = range(len(y))
        for x, y, text in zip(x, y, label_markers):
            plt.text(x, y, text)
        

    
def q0_star_mean_scatter(df):
    plt.figure()
    episodes = df.groupby('ep_id')
    q0_star = []
    a0_star = []
    q0_mean = []
    for i, ng_tuple in enumerate(episodes):
        name, group = ng_tuple 
        s0_all_q = get_qvals_at_idx(group,0)
        q0_star.append(np.max(s0_all_q))
        a0_star.append(np.argmax(s0_all_q))
        q0_mean.append(np.mean(s0_all_q))
        
    # plt.scatter(range(len(episodes)),q0_star, label='q*(s0)')
    # plt.scatter(range(len(episodes)),a0_star, label='a*(s0)')
    # plt.scatter(range(len(episodes)),q0_mean, label='q-mean(s0)')
    plt.plot(range(len(episodes)),q0_star,marker='s', label='q*(s0)',markersize=2)
    plt.plot(range(len(episodes)),a0_star,marker='s', label='a*(s0)',markersize=2)
    plt.plot(range(len(episodes)),q0_mean,marker='s', label='q-mean(s0)',markersize=2)
    
    plt.legend()
    
def qt_with_var(df):
    episodes = df.groupby('ep_id')
    plt.figure()
    for i, ng_tuple in enumerate(episodes):
        name, group = ng_tuple 
        qvals_col = parse_qvals_col(group)
        qt_mean = qvals_col.apply(lambda x: np.mean(x))
        qt_min = qvals_col.apply(lambda x: np.min(x))
        qt_star = qvals_col.apply(lambda x: np.max(x))
        # qt = qvals_col[group['at_id']]
        # plt.scatter(group.index,qt_mean,linewidths=0.001)
        # plt.scatter(group.index,qt_star,linewidths=0.001)
        # plt.scatter(group.index,qt_min,linewidths=0.001)
        plt.plot(qt_mean)
        plt.plot(qt_min)
        plt.plot(qt_star)
    plt.title('q(t) min, q(t) avg, q*(t) ')



def reward_sum_with_labels(df):
    episodes = df.groupby('ep_id')
    plt.figure()
    x = range(len(episodes))
    labels = [str(i) for i in x]
    y = []
    for i, ng_tuple in enumerate(episodes):
        name, group = ng_tuple 
        y.append(np.sum(group['rt']))   
    plt_with_label_markers(x,y,labels)
        

# def live_plot():
#     fig = plt.figure()
#     from numpy import cos
#     plt.ion()
#     for i in range(100000):
#         fig.clear()
#         plt.plot(range(i),np.cos(range(i)))
#         plt.pause(0.0001) # Gi

# live_plot()     
        
        
        

    # qt_var = qvals_col.apply(lambda x: np.var(x))
    # qt = qvals_col[df['at_id']]
    # qt_star = qvals_col.apply(lambda x: np.max(x))    
    # plt.errorbar(df.index, qt_mean, qt_var, fmt="o", color="r")

def rt_ep_sum_bar(df):
    episodes = df.groupby('ep_id')
    plt.figure()
    q0_star = []
    rsums = []
    for i, ng_tuple in enumerate(episodes):
        name, group = ng_tuple 
        # s0_all_q = get_qvals_at_idx(group,0)
        rsums.append(np.sum(group['rt']))
        # plt.bar(group.index, np.sum(group['rt']))
    plt.title('reward sum over episodes')
    plt.bar(range(len(rsums)), rsums)

def episode_losses_bar(df):
    episodes = df.groupby('ep_id')
    plt.figure()
    mean_losses = []
    for i, ng_tuple in enumerate(episodes): 
        if i == 0: # here we remove the first episode since its not computing loss from the first time step
            continue
        name, group = ng_tuple 
        # s0_all_q = get_qvals_at_idx(group,0)
        mean_losses.append(np.mean(group['optimMDloss']))
        # plt.bar(group.index, np.sum(group['rt']))
    plt.title('episode mean loss per step')
    plt.bar(range(len(mean_losses)), mean_losses)

def episode_action_changing_freq_bar(df):
    episodes = df.groupby('ep_id')
    plt.figure()
    change_frequency = [] # times per action
    for i, ng_tuple in enumerate(episodes):
        name, group = ng_tuple 
        # s0_all_q = get_qvals_at_idx(group,0)
        changes = group['at_id'].diff() != 0 
        change_cnt = changes.sum() # num of action changes in episode
        change_frequency.append(change_cnt/len(group.index)) # divided by length of episode 
    plt.bar(range(len(episodes)),change_frequency)
    plt.title('episode action change frequency (total episode action changes[no unit]\n/ episode length[ts])')

def step_times_grouped_plt(df):
    episodes = df.groupby('ep_id')
    fig =plt.figure()
    episode_step_times = [] # times per action
    for i, ng_tuple in enumerate(episodes):
        name, group = ng_tuple 
        group = group[1:] # removing the first observation since it always had a very large step time (not reflecting the data)
        step_time = group['stepMDduration'] 
        plt.plot(step_time)
    
    plt.xlabel('t')
    plt.title('step time (color = episode)')


def episode_mean_step_time(df):
    episodes = df.groupby('ep_id')
    fig =plt.figure()
    episode_step_time_mean = [] # times per action
    for i, ng_tuple in enumerate(episodes):
        name, group = ng_tuple 
        group = group[1:] # removing the first observation since it always had a very large step time (not reflecting the data)
        step_time = group['stepMDduration']
         
        episode_step_time_mean.append(step_time.mean())
    plt.bar(range(len(episodes)),episode_step_time_mean)
    plt.xlabel('episode number')
    plt.title('mean step time over episode')

def action_feature_grouped(df,feature_name):
    episodes = df.groupby('ep_id')
    fig =plt.figure()
    episode_step_time_mean = [] # times per action
    for i, ng_tuple in enumerate(episodes):
        name, group = ng_tuple 
        action = group[feature_name] 
        plt.plot(action)
        # plt.scatter(group.index, action,linewidths=0.01)
    plt.xlabel('t')
    plt.title(f'{feature_name}')

# rt_bar(df)
# q0_star_plot(df)
rt_ep_sum_bar(df)
rt_plot_grouped(df)
q0_star_mean_scatter(df)
episode_losses_bar(df)
episode_action_changing_freq_bar(df)
episode_mean_step_time(df)
step_times_grouped_plt(df)
action_feature_grouped(df,'at_particles')
qt_with_var(df)
# reward_sum_with_labels(df)
# print(df.value_counts())
###########
###########
plt.show()
##########

# time.sleep(100)

# x = np.arange(len(y))
# plt.plot(x,y)
# plt.show()
 