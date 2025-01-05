import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# matplotlib.use('Agg')  # Use a non-interactive backend
df = pd.read_csv('/home/dan/MPC_RLPT/BGU/Rlpt/trained_models/2025:01:05(Sun)12:16:36/etl.csv')
# y = np.arange(10)
# plt.plot(y)
# plt.show()
# y = df['optim_raw_grad_norm']


#########
max_episode = 300
episodes = df.groupby('ep')
for i, ng in enumerate(episodes):
    name, group = ng
    if i < max_episode:
        plt.plot(group['q(w,st,at)'])
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
print(episodes)
last_episode = episodes.get_group(episodes.ngroups-1) # last episode
q = last_episode['q(w,st,at)']
pos_err = last_episode['pos_er_s(t+1)']
rot_err = last_episode['rot_er_s(t+1)']
# plt.plot(y=q,x=[pos_err, rot_err])
 
# syntax for 3-D projection
ax = plt.axes(projection ='3d')
 
# defining axes
z = q
x = pos_err
y = rot_err
# c = x + y
# ax.scatter(x, y, z, c = c)
ax.scatter(x, y, z)
ax.set_title('q(pos_err, rot_err) in last episode')
ax.set_xlabel('pos_err', fontsize=12)
ax.set_ylabel('rot_err', fontsize=12)
plt.show()

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
