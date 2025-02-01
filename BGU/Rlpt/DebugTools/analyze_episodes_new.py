from cProfile import label
import copy
import matplotlib
# matplotlib.use('Agg')  # Use a non-interactive backend

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from tomlkit import item
# path = '/home/dan/MPC_RLPT/BGU/Rlpt/favorite_models/2025:01:25(Sat)18:56:12/training_etl.csv'
# path = '/home/dan/MPC_RLPT/BGU/Rlpt/favorite_models/2025:01:25(Sat)23:05:28/training_etl.csv'
# path = '/home/dan/MPC_RLPT/BGU/Rlpt/trained_models/2025:01:27(Mon)19:17:42/training_etl.csv' # storm original
# path = '/home/dan/MPC_RLPT/BGU/Rlpt/trained_models/2025:01:27(Mon)19:29:57/training_etl.csv' # storm original
# path = 'Rlpt/favorite_models/2025:01:28(Tue)00:04:53/training_etl.csv'
# path = '/home/dan/MPC_RLPT/BGU/Rlpt/favorite_models/2025:01:30(Thu)00:11:49_v1092/training_etl.csv'
# path = '/home/dan/MPC_RLPT/BGU/Rlpt/favorite_models/2025:01:30(Thu)17:10:19_v1235/training_etl.csv'
path = '/home/dan/MPC_RLPT/BGU/Rlpt/favorite_models/2025:01:31(Fri)13:14:00_v1486/training_etl.csv'
df = pd.read_csv(path)
print(df.head)
print(df.nunique())


    
def parse_st_pi_mppi_means(df):
    df['st_pi_mppi_means'] = df['st_pi_mppi_means'].apply(lambda x: x[0]) 
    return df
# df_parsed = parse_st_pi_mppi_means(df)

    
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

def ee_errors(df):
    pos_err = 's(t+1)MDpos_err'
    rot_err = 's(t+1)MDrot_err'
    episodes = df.groupby('ep_id')
    plt.figure()
    for i, ng_tuple in enumerate(episodes):
        name, group = ng_tuple 
        plt.plot(group[pos_err])   
        plt.plot(group[rot_err])   
    plt.title('ee errors (pos, rot)')
    plt.show()

def reward_sum_with_labels(df):
    episodes = df.groupby('ep_id')
    plt.figure()
    x = range(len(episodes))
    labels = [str(i) for i in x]
    y = []
    for i, ng_tuple in enumerate(episodes):
        name, group = ng_tuple 
        y.append(np.sum(group['rt']))   
    plt.plot(y)
    plt.title('episode total reward')
    # plt_with_label_markers(x,y,labels)




def rt_ep_sum_bar(df):
    episodes = df.groupby('ep_id')
    plt.figure()
    q0_star = []
    rsums = []
    r_accumulate = [0]
    for i, ng_tuple in enumerate(episodes):
        name, group = ng_tuple 
        # s0_all_q = get_qvals_at_idx(group,0)
        rsums.append(np.sum(group['rt']))
        # plt.bar(group.index, np.sum(group['rt']))
        if i == 0:
            r_accumulate.append(rsums[0])
        else:
            r_accumulate.append(r_accumulate[-1] + rsums[i])
    plt.title('reward sum over episodes')
    plt.bar(range(len(rsums)), rsums)
    # plt.plot(r_accumulate)
    

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
    
def action_index(df):
    episodes = df.groupby('ep_id')
    plt.figure()
    plt.xlabel('t')
    plt.ylabel(f'action index')
    # plt.plot(df['at_id'])
    for i, ng_tuple in enumerate(episodes):
        name, group = ng_tuple 
        action = group['at_id'] 
        # plt.scatter(group.index, action)
        a0 = action[action == 0]
        a1 = action[action == 1]
        a2 = action[action == 2]
        plt.scatter(a0.index, a0, c='red', linewidths=0.001)
        plt.scatter(a1.index, a1, c='blue',linewidths=0.001)
        plt.scatter(a2.index,a2, c='green',linewidths=0.001)
        
        
def qsa_all(df,ep_id):
    plt.figure()
    ep = df[df['ep_id'] == ep_id]
    qvals = parse_qvals_col(ep)
    qsa_by_a = []
    n_actions = df['at_id'].nunique()
    for i in range(n_actions):
        q_action = qvals.apply(lambda x:x[i])
        qsa_by_a.append(q_action)
    ai_plots = []
    ai_colors = [] 
    for i in range(n_actions):
        p_ai = plt.plot(qsa_by_a[i], label = f'q action {i}',linewidth=1)
        ai_plots.append(p_ai)
        ai_colors.append(p_ai[0].get_color())
    plt.plot(ep['rt'],label='reward')
    plt.plot(ep['at_id'], label='taken action')
    df_qs = pd.DataFrame(columns=list(range(n_actions)))
    for i in range(n_actions):
        df_qs[i] = qsa_by_a[i]
    df_greedy = df_qs.idxmax(axis=1)
    plt.plot(df_greedy, label='greedy action')
    
    greedy_action_list = list(df_greedy)
    greedy_action_list = [int(a_id) for a_id in greedy_action_list]
    
    for i in ep.index:
        plt.axvspan(i, i + 1, facecolor=ai_colors[greedy_action_list[i - ep.index[0]]], alpha=0.5)
    plt.title(f'color background = greedy action, episode = {ep_id}')
    plt.legend()
    
def reward_sum(df):
    plt.figure()
    rewards = df['rt']
    rsums = [0]
    for i in range(1,len(rewards.index)):
        rsums.append(rsums[i-1] + df['rt'][i])
    plt.plot(rsums,label='reward sum')
    plt.plot(df['rt'],label='rt')
    plt.title('rewards over training')
    plt.legend()  


def cor_action_traj_return(df):
    episodes = df.groupby('ep_id')
    trajectories = []
    returns = []
    a = np.array()
    for i, ng_tuple in enumerate(episodes):
        name, group = ng_tuple     
        traj = group['at_id'] 
        g =  group['rt'].sum()
        trajectories.append(np.array(traj))
        returns.append(g)
        a = np.stack((a,traj), axis=0)
        
    # X = np.array(trajectories)  # Shape: (num_episodes, n)
    Y = np.array(returns)
    
    mi = mutual_info_regression(a,Y)
    print('mi = ', mi)

def ep_dur(df):
    plt.figure()
    episodes = df.groupby('ep_id')
    t_steps = []
    durs = []
    for i, ng_tuple in enumerate(episodes):
        name, group = ng_tuple     
        n_steps = group['t_ep'][group.index[-1]]# len(group)
        dur = sum(group['stepMDduration'])
        t_steps.append(n_steps)
        durs.append(dur)
        
    plt.plot(t_steps,label='t_steps')
    plt.plot(durs,label='duration')
    plt.legend()
    
def split_pose(x):
    x = x[1:-1]
    x = x.split('  ')
    
def pos(df):
    plt.figure()
    episodes = df.groupby('ep_id')
    fig = plt.figure()
    for i, ng_tuple in enumerate(episodes):
        ax = plt.axes(projection ='3d')
        name, group = ng_tuple
        
        # group['s(t+1)MDee_pose_gym_cs'] = group['s(t+1)MDee_pose_gym_cs'].str[1:-1].split(',').join() 
        # group['s(t+1)MDee_pose_gym_cs_parsed'] = group['s(t+1)MDee_pose_gym_cs'].str[1:-1].str.split('  ')
        # group['s(t+1)MDee_pose_gym_cs_parsed'] = group['s(t+1)MDee_pose_gym_cs_parsed'].apply(split_pose)
        # .apply(lambda x: [float(s) for s in x])
        group['pos_x'] = group['s(t+1)MDee_pose_gym_cs'].apply(lambda x: [0])     
        group['pos_y'] = group['s(t+1)MDee_pose_gym_cs'].apply(lambda x: x[1])
        group['pos_z'] = group['s(t+1)MDee_pose_gym_cs'].apply(lambda x: x[2])
        ax.plot(group['pos_x'].values, group['pos_y'].values, group['pos_z'], alpha=0.5,label=f'episode {name}: {str(label)}') # simple blue line
        
# cor_action_traj_return(df)
    


# qsa_all(df, 0)
# qsa_all(df, 100)
# qsa_all(df, 200)
# qsa_all(df, 300)
# qsa_all(df, 400)
# qsa_all(df, 500)
# qsa_all(df, 600)
# qsa_all(df, 700)
# qsa_all(df, 701)
# qsa_all(df, 800)
# qsa_all(df, 801)
# qsa_all(df, 822)

# qsa_all(df, 918)
# qsa_all(df, 919)
# qsa_all(df, 920)
# qsa_all(df, 921)
# qsa_all(df, 1090)
# qsa_all(df, 1091)

# qsa_all(df,1230)
# qsa_all(df,1231)
# qsa_all(df,1232)
# qsa_all(df,1233)
# qsa_all(df,1234)
# qsa_all(df,1235)
# pos(df[df['ep_id'] == 334])
# qsa_all(df,566)
# qsa_all(df,567)
qsa_all(df,1401) # terminal
qsa_all(df,1482)
qsa_all(df,1483)
qsa_all(df,1484)
qsa_all(df,1485)
qsa_all(df,1486)




# ep_dur(df)



# success traces
# action_index(df[df['ep_id'] == 79])
# action_index(df[df['ep_id'] == 93])
# action_index(df[df['ep_id'] == 140])
# action_index(df[df['ep_id'] == 141])


# rt_bar(df)
# q0_star_plot(df)
rt_ep_sum_bar(df)
# rt_plot_grouped(df)
# q0_star_mean_scatter(df)
episode_losses_bar(df)
# episode_action_changing_freq_bar(df)
# episode_mean_step_time(df)
# step_times_grouped_plt(df)
# # action_feature_grouped(df,'at_particles')
# action_feature_grouped(df,'at_particles')
# action_index(df)
# qt_with_var(df)
# ee_errors(df)
# reward_sum_with_labels(df)
# print(df.value_counts())
reward_sum_with_labels(df)
reward_sum(df)
###########
plt.show()
##########


# x = np.arange(len(y))
# plt.plot(x,y)
# plt.show()
 