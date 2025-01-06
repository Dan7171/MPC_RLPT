import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# matplotlib.use('Agg')  # Use a non-interactive backend
# path = '/home/dan/MPC_RLPT/BGU/Rlpt/trained_models/2025:01:05(Sun)12:16:36/etl.csv' 760 episodes, with pushing truncated into buffer
path = '/home/dan/MPC_RLPT/BGU/Rlpt/trained_models/2025:01:06(Mon)13:35:23/etl.csv' # 194 episodes, without pushing truncated into buffer
df = pd.read_csv(path)
# y = np.arange(10)
# plt.plot(y)
# plt.show()
# y = df['optim_raw_grad_norm']

print(df.head)
print('columns:')
print(df.columns)
print(df.nunique())
# df['rt'].value_counts()
#########
max_episode = 800
episodes = df.groupby('ep')
base_color = 'orange'
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

#########
plt.figure()
q_start = []
for i, ng in enumerate(episodes):
    name, group = ng
    if i < max_episode:
        first_q = group['q(w,st,at)'][group.index[0]]
        q_start.append(first_q)
plt.plot(q_start)
plt.title(f'first {min(i+1,max_episode)} episodes (each color = episode) q start')

###########
fig = plt.figure()
last_episode = episodes.get_group(episodes.ngroups-1) # last episode
q = last_episode['q(w,st,at)']
pos_err = last_episode['pos_er_s(t+1)']
rot_err = last_episode['rot_er_s(t+1)']
# defining axes
ax = plt.axes(projection ='3d') 
z = q
x = pos_err
y = rot_err
# ax.scatter(x, y, z) # also works (but one color)
c = x + y # changes color, but not a must
ax.scatter(x, y, z, c = c)
ax.set_title('q(pos_err, rot_err) in last episode')
ax.set_xlabel('pos_err', fontsize=12)
ax.set_ylabel('rot_err', fontsize=12)

#############
fig = plt.figure()
for i, ng in enumerate(episodes):
    name, group = ng
    if i < max_episode:
        plt.plot(group['optim_raw_grad_norm'])
plt.title(f'first {min(i+1,max_episode)} episodes (each color = episode) norm of minibatch gradient at optimization step (unclipped)')

#############
fig = plt.figure()
for i, ng in enumerate(episodes):
    name, group = ng
    if i < max_episode:
        plt.plot(group['optim_loss'])
plt.title(f'first {min(i+1,max_episode)} episodes (each color = episode) total minibatch loss at optimization step')



#############
fig = plt.figure()
for i, ng in enumerate(episodes):
    name, group = ng
    if i < max_episode:
        plt.plot(group['at_epsilon'])
plt.title(f'first {min(i+1,max_episode)} episodes (each color = episode) random action chance')





###########
plt.show()
##########
# plt.show()
# for name, group in df:
#     plt.plot(group.index, group['q(w,st,at)'], label=f'Group {name}')
# y = df['q(w,st,at)']
# # y = df['at_particles']
# # # y.fillna(value=0)
# # x = np.arange(len(y))
# # plt.plot(x,y)
# print(y.unique())
# y.
#  plt.plot()



# time.sleep(100)

# x = np.arange(len(y))
# plt.plot(x,y)
# plt.show()
