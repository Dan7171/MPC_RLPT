import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tomlkit import item


def get_dofs_over_time(df):
    dofs_parsed_df = df['st_robot_dofs_positions'].apply(lambda x: np.fromstring(x.strip("[]"), sep=" ")) # type: ignore
    dofs_parsed_np = np.stack(dofs_parsed_df.to_numpy())
    return dofs_parsed_np


DOF_STATE_SIZE = 7 
GRID_DIM_LEN = 5 # determines grid granularity


x = np.arange(DOF_STATE_SIZE, dtype=np.float32)
dofs_min_vals = np.full_like(x, np.inf)
dofs_max_vals = np.full_like(x, -np.inf)
print(dofs_min_vals)
print(dofs_max_vals)



etls = ['/home/dan/MPC_RLPT/BGU/Rlpt/favorite_models/2025:01:10(Fri)20:41:09___128_start_actions(no_tuning)_for_pca/training_etl.csv']

# calc grid centers
for path in etls: # each path will contain E episodes with same parameter set (tunable), which performed random actions
    df = pd.read_csv(path)
    dofs_parsed_np = get_dofs_over_time(df)
    dofs_min_vals = np.minimum(dofs_min_vals, np.min(dofs_parsed_np[:],axis=0))
    dofs_max_vals = np.maximum(dofs_max_vals,np.max(dofs_parsed_np[:],axis=0))
    # print(dofs_min_vals)
    # print(dofs_max_vals)
    
# print('axis list')
axis_centers_list = [] 
for i in range(DOF_STATE_SIZE):
    axis_centers_list.append(np.linspace(dofs_min_vals[i], dofs_max_vals[i], GRID_DIM_LEN))
    # print(axis_centers_list[-1])
centers_template_matrix = np.column_stack(axis_centers_list)
centers_matrix = np.array(list(itertools.product(*centers_template_matrix.T))) # all combinations in templet
# print(centers_matrix.shape)  # This will be (GRID_DIM_LEN^DOF_STATE_SIZE, DOF_STATE_SIZE)

for path in etls:
    df = pd.read_csv(path)
    dofs_parsed_np = get_dofs_over_time(df)
    coverage_histogram = np.zeros_like(np.arange(len(centers_matrix)))
    for p in dofs_parsed_np:
        distances = np.linalg.norm(centers_matrix - p, axis=1)   
        # Find the index of the minimum distance
        min_index = np.argmin(distances)
        # Retrieve the closest point p'
        p_prime = centers_matrix[min_index]
        # print(f'p = {p}, closest = {p_prime}, dist = {np.min(distances)}')
        coverage_histogram[min_index] += 1
    pass