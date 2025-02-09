from cProfile import label
import copy
import matplotlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from tomlkit import item
    
def _parse_qvals(qvals_str):
    qvals_list = qvals_str[1:-1].split(', ')
    qvals_list = [float(x) for x in qvals_list]
    return qvals_list

def parse_qvals_col(df):
    qvals_col = df['atMDq(w,st,all)']
    qvals_col = qvals_col.apply(lambda x: _parse_qvals(x))
    return qvals_col
    
def qsa_all(df,ep_id):
    ep = df[df['ep_id'] == ep_id]
    qvals = parse_qvals_col(ep)
    qsa_by_a = []
    n_actions = df['at_id'].nunique()
    for i in range(n_actions):
        q_action = qvals.apply(lambda x:x[i])
        qsa_by_a.append(q_action)
    
    df_qs = pd.DataFrame(columns=list(range(n_actions)))    
    for i in range(n_actions):
        df_qs[i] = qsa_by_a[i]
    return df_qs

    # df_greedy = df_qs.idxmax(axis=1)
    # plt.plot(df_greedy, label='greedy action')
    
    # greedy_action_list = list(df_greedy)
    # greedy_action_list = [int(a_id) for a_id in greedy_action_list]
    
    # for i in ep.index:
    #     plt.axvspan(i, i + 1, facecolor=ai_colors[greedy_action_list[i - ep.index[0]]], alpha=0.5)
    # plt.title(f'color background = greedy action, episode = {ep_id}')
    # plt.legend()
    


def calc_true_values(df):
    gamma = 0.999
    episode_0 = df[df['ep_id'] == 0]
    true_rewards = list(episode_0['rt'])
    assert len(true_rewards) == 156
    true_values = [0 for i in range(len(true_rewards))]
    for i in range(len(true_values)-2 , -1, -1):
        # print(true_rewards[i], gamma * true_values[i+1] )
        true_values[i] = true_rewards[i] + gamma * true_values[i+1] 
    
    return true_values

def check_routine(check_path, check_id, check_reward_type):
    print(f"routine for check {check_id} starts")
    df = pd.read_csv(check_path)
    print(df.head)
    print(df.nunique())
    
    if check_id == 1 or check_id == 2 or check_id == 3 or check_id == 4:
        true_v = calc_true_values(df) # since the 2 possible actions actions are the same, all  episodes are deterministic and the rewards remain the same along every episode
        n_episodes = len(df.groupby('ep_id'))
        q_table_last_episode = qsa_all(df, n_episodes-1)
        n_actions = q_table_last_episode.shape[1]
        converged_v = list(q_table_last_episode.sum(axis=1) / n_actions)
        plt.figure()
        plt.title(f'check {check_id} actual converged V vs real V. Reward = {check_reward_type}')
        plt.plot(true_v, label='true value function')
        plt.plot(converged_v, label=f'converged value function (at final episode no {n_episodes})')
        plt.legend()

path_check1 = '/home/dan/MPC_RLPT/BGU/Rlpt/favorite_models/2025:02:07(Fri)23:49:54/training_etl.csv'
path_check2 =  '/home/dan/MPC_RLPT/BGU/Rlpt/favorite_models/2025:02:08(Sat)09:24:43/training_etl.csv'
path_check3 =  '/home/dan/MPC_RLPT/BGU/Rlpt/favorite_models/2025:02:08(Sat)13:52:52/training_etl.csv'
path_check4 =  '/home/dan/MPC_RLPT/BGU/Rlpt/favorite_models/2025:02:08(Sat)21:54:45/training_etl.csv'
checks_reward_funcs = ['-pose error', '-pose error', '-1', '-1']
checks_paths = [path_check1, path_check2, path_check3, path_check4]
for i,p in enumerate(checks_paths):
    check_routine(p, i+1, checks_reward_funcs[i])
plt.show()