from cProfile import label
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tomlkit import item


def pca3d(ndim_variable_name,max_episodes_in_on_figure=8):
    
    ndim_var_parsed_df = df[ndim_variable_name].apply(lambda x: np.fromstring(x.strip("[]"), sep=" "))
    ndim_var_parsed_n = np.stack(ndim_var_parsed_df.to_numpy())
    pca = PCA(n_components=2)  # For 2D
    data_2d = pca.fit_transform(ndim_var_parsed_n)
    # Plot the 2D projection
    x = data_2d[:, 0]
    y = data_2d[:, 1]
    z = range(len(data_2d))
    

    axs = []
    figs = []
    groupnum = 0
    for _,group in episodes:
        if groupnum % max_episodes_in_on_figure == 0:
            fig = plt.figure()
            ax = plt.axes(projection ='3d')
            figs.append(fig)
            axs.append(ax)
        df_group = pd.DataFrame(index=group.index,columns=['x', 'y'])    
        df_group['x'] = x[df_group.index]
        df_group['y'] = y[df_group.index]
        z = df_group.index # timestep
        
        label = []
        for k, n_unique in df_varying_actions.items():
            label.append(group[k][group.index[0]])
        
        axs[-1].scatter(df_group['x'],df_group['y'],z, alpha=0.5 ,s=10, linewidths=0.1,label=f'episode {groupnum}: {str(label)}')         
        
        if groupnum % (max_episodes_in_on_figure - 1)  == 0: #  == 0 or groupnum == len(episodes)-1:
            
            axs[-1].legend(title=str(list(df_varying_actions.keys())),fontsize='x-small') # axs[-1].legend(fontsize='x-small',title=str(list(df_varying_actions.keys())))
            axs[-1].set_xlabel(f"x = {ndim_variable_name} PC1")
            axs[-1].set_ylabel(f"y = {ndim_variable_name} PC2")
            axs[-1].set_title('z = time step (t)')
            figs[-1].tight_layout()


        groupnum += 1
        
 

        

    

#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
##################################### START HERE ########################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################
#########################################################################################



# matplotlib.use('Agg')  # Use a non-interactive backend
# path = '/home/dan/MPC_RLPT/BGU/Rlpt/trained_models/2025:01:05(Sun)12:16:36/etl.csv' 760 episodes, with pushing truncated into buffer
# path = "/home/dan/MPC_RLPT/BGU/Rlpt/trained_models/2025:01:06(Mon)22:47:44/etl.csv"# '/home/dan/MPC_RLPT/BGU/Rlpt/trained_models/2025:01:06(Mon)13:35:23/etl.csv' # 194 episodes, without pushing truncated into buffer
# path = '/home/dan/MPC_RLPT/BGU/Rlpt/trained_models/2025:01:10(Fri)17:29:59/training_etl.csv'
# path = '/home/dan/MPC_RLPT/BGU/Rlpt/trained_models/2025:01:10(Fri)20:08:00/training_etl.csv'
# path = '/home/dan/MPC_RLPT/BGU/Rlpt/trained_models/2025:01:10(Fri)20:29:59/training_etl.csv'
# path = '/home/dan/MPC_RLPT/BGU/Rlpt/favorite_models/2025:01:10(Fri)20:41:09___128_start_actions(no_tuning)_for_pca/training_etl.csv'
path = '/home/dan/Desktop/training_etl.csv' # 2025:01:21(Tue)01:37:25 with HER and 16 actions
df = pd.read_csv(path)
# y = np.arange(10)
# plt.plot(y)
# plt.show()
# y = df['optim_raw_grad_norm']

print(df.head)
print('columns:')
print(df.columns)
print(df.nunique())
df_nunique = df.nunique().to_dict()
# df['rt'].value_counts()

max_episode = 1500
episodes = df.groupby('ep')
# %%%%%%%%%%%%% FIG 1 %%%%%%%%%%%%%%%

base_color = 'orange'

# GROUPED BY EPISODES (FOR COLOR SEPARATION)
for i, ng in enumerate(episodes):
    name, group = ng
    print(f'{i}, {len(group)}')


# FIG 1: Q values  
# Red episodes: colision
# Green episodes: goal
# Orange: else
for i, ng in enumerate(episodes):
    name, group = ng
    if i < max_episode:

        contact_q = pd.DataFrame(index=group.index,columns=['q_if_contact'],dtype=float)
        contact_q[:] = 0
        contact_q.loc[group['contact_s(t+1)'], 'q_if_contact'] = group['q(w,st,at)']
        contact_q = contact_q[contact_q['q_if_contact'] !=0 ]
        plt.plot(contact_q['q_if_contact'],'X')
        
        goal_reached_q = pd.DataFrame(index=group.index,columns=['q_if_goal_reached'],dtype=float)
        goal_reached_q[:] = 0
        goal_reached_q.loc[group['goal_reached_s(t+1)'], 'q_if_goal_reached'] = group['q(w,st,at)']
        goal_reached_q = goal_reached_q[goal_reached_q['q_if_goal_reached'] !=0 ]
        plt.plot(goal_reached_q['q_if_goal_reached'], '+') # linewidth='2.0')
        
        if len(contact_q):
            q_color='red'
        elif len(goal_reached_q):
            q_color='green'
        else:
            q_color = base_color
        plt.plot(group['q(w,st,at)'],color=q_color)
        
plt.title(f'first {min(i+1,max_episode)} episodes (each color = episode) q vals')


# %%%%%%%%%%%%% FIG 2 %%%%%%%%%%%%%%%
# FIG 2: Q(s0,a0) for each episode  
#########
plt.figure()
q_start = []
for i, ng in enumerate(episodes):
    name, group = ng
    if i < max_episode:
        first_q = group['q(w,st,at)'][group.index[0]]
        q_start.append(first_q)
plt.plot(q_start)
plt.title(f'first {min(i+1,max_episode)} episodes "Q0". X = episode num, Y = q(s0,a0)')

# %%%%%%%%%%%%% FIG 3 %%%%%%%%%%%%%%%
###########
# FIG 3: Last epidode data: x = pos error, y = rot error, z = q value. Each point (x,y,z) represents one time step, color gradient: the more its yellow, the earlier in episode it is  

fig = plt.figure()
last_episode_group_num = episodes.ngroups-1
last_episode = episodes.get_group(last_episode_group_num) # last episode
q = last_episode['q(w,st,at)']
pos_err = last_episode['pos_er_s(t+1)']
rot_err = last_episode['rot_er_s(t+1)']



# set 3d axis
x = pos_err
y = rot_err
z = q

# defining axes
ax = plt.axes(projection ='3d') 

# ax.scatter(x, y, z) # also works (but one color)
c = x + y # changes color, but not a must
ax.set_title('q(pos_err, rot_err) in last episode')
ax.set_xlabel('pos_err', fontsize=12)
ax.set_ylabel('rot_err', fontsize=12)

ax.plot(x.values, y.values, z.values) # simple blue line
ax.scatter(x, y, z, c = c) # adds marking points with color gradient to simple line




# %%%%%%%%%%%%% FIG 4 %%%%%%%%%%%%%%%
# FIG 4: Raw Gradient norm of update step over entire minibache for "max_episode" first episodes in data (color = episode)
#############
if df_nunique['optim_raw_grad_norm'] > 1:    
    fig = plt.figure()
    for i, ng in enumerate(episodes):
        name, group = ng
        if i < max_episode:
            plt.plot(group['optim_raw_grad_norm'])

    plt.title(f'first {min(i+1,max_episode)} episodes (each color = episode) norm of minibatch gradient at optimization step (unclipped)')


# %%%%%%%%%%%%% FIG 5 %%%%%%%%%%%%%%%
# FIG 5: Loss of update step over entire minibache for "max_episode" first episodes in data (color = episode)
#############

if df_nunique['optim_loss'] > 1:
    fig = plt.figure()
    for i, ng in enumerate(episodes):
        name, group = ng
        if i < max_episode:
            plt.plot(group['optim_loss'])
    plt.title(f'first {min(i+1,max_episode)} episodes (each color = episode) total minibatch loss at optimization step')



# %%%%%%%%%%%%% FIG 6 %%%%%%%%%%%%%%%
# FIG 6: Epsilon value of action over all actions in first max_episdoe episodes (color = episode) 
#############
fig = plt.figure()
for i, ng in enumerate(episodes):
    name, group = ng
    if i < max_episode:
        plt.plot(group['at_epsilon'])
plt.title(f'first {min(i+1,max_episode)} episodes (each color = episode) random action chance')



# # %%%%%%%%%%%%% FIGS 7 to 7+"K1"+"K2" (PCA figures of dof states and mpc policy) %%%%%%%%%%%%%%%
# ####### TODO - AT THE MOMENT - THE LEGEND REPRESENTS THE FIRST ACTION IN EPISODE ONLY) 
# df_varying_actions = {k:df_nunique[k] for k in df_nunique if (k.startswith('at_') and (not k.startswith('at_dur')) and df_nunique[k] > 1) }

# # NEXT "K1" FIGURES: FIGS 7 to 7+K1 (K1 determined dinamically based on num of episodes): PCA over 2 first components in st_robot_dofs_positions (robot joints state) at time t, each figure represents a subset of episoides, x = first pca, y = second pca, z = time step, (color = episode) 
# pca3d('st_robot_dofs_positions') 
# # NEXT "K2" FIGURES: FIGS 7+K1 to 7+K1+K2 PCA over 2 first components in MPC policy (means only, no standard devs) at time t, each figure represents a subset of episoides, x = first pca, y = second pca, z = time step, (color = episode)
# pca3d('st_pi_mppi_means')




###########
###########
plt.show()
##########

# time.sleep(100)

# x = np.arange(len(y))
# plt.plot(x,y)
# plt.show()
