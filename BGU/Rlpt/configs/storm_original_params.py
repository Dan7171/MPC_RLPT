
cost_fn_space_default = {  # for the original params see: storm/content/configs/mpc/franka_reacher.yml
    "goal_pose":  [(15.0, 100.0)],     # distance from goal pose (orientation err weight, position err weight).
    "zero_vel": [0.0], 
    "zero_acc": [0.0],
    "joint_l2": [0.0], 
    "robot_self_collision": [5000], # collision with self (robot with itself)
    "primitive_collision" : [5000], # collision with environment (obstacles) 
    "voxel_collision" : [0.0],
    "null_space": [1.0],
    "manipulability": [30], 
    "ee_vel": [0.0], 
    "stop_cost" : [(100.0, 1.5)], # charging for crossing max velocity limit during rollout (weight, max_nlimit (max acceleration)) # "stop_cost": [(100.0, 1.5), # high charging (100), and low acceleration limit (1.5) 
    "stop_cost_acc": [(0.0, 0.1)],# charging for crossing max acceleration limit (weight, max_limit)
    "smooth": [1.0], # smoothness weight
    "state_bound": [1000.0] # joint limit avoidance weight
    }

mppi_space_default = {
    'horizon': [30],
    'n_iters': [1],
    'particles': [500]
}


# cost_fn_space_defaults = {
#     "goal_pose": {'weight': [15.0, 100.0]# orientation, position.
#                     }, 
#     "zero_vel": {'weight': 0.0}, 
#     "zero_acc": {'weight': 0.0},
#     "joint_l2": {'weight': 0.0},
#     "robot_self_collision": {'weight': 5000,
#                                 'distance_threshold': 0.05
#                                 },
#     "primitive_collision" : {'weight': 5000,
#                                 'distance_threshold': 0.05
#                                 },
#     "voxel_collision" :{'weight': 0.0},
#     "null_space": {'weight': 1.0},
#     "manipulability": {'weight': 30},
#     "ee_vel": {'weight': 0.0}, 
#     "stop_cost": {'weight': 100,
#                     'max_nlimit': 1.5 # maximal joint speed acceleration on some a(t,h)
#                     },         
#     "stop_cost_acc": {'weight': 0.0,
#                         'max_limit': 0.1}, # ? 
#     "smooth": {'weight': 0.0}, # smoothness
#     "state_bound": {'weight': 1000}, # joint limit avoidance
# }
# mppi_space_defaults = {'horizon': 30,
#                         'n_iters': 1,
#                         'num_particles': 500
#                     }

