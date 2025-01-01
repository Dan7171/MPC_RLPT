
cost_fn_space_defaults = {
    "goal_pose": {'weight': [15.0, 100.0]# orientation, position.
                    }, 
    "zero_vel": {'weight': 0.0}, 
    "zero_acc": {'weight': 0.0},
    "joint_l2": {'weight': 0.0},
    "robot_self_collision": {'weight': 5000,
                                'distance_threshold': 0.05
                                },
    "primitive_collision" : {'weight': 5000,
                                'distance_threshold': 0.05
                                },
    "voxel_collision" :{'weight': 0.0},
    "null_space": {'weight': 1.0},
    "manipulability": {'weight': 30},
    "ee_vel": {'weight': 0.0}, 
    "stop_cost": {'weight': 100,
                    'max_nlimit': 1.5 # maximal joint speed acceleration on some a(t,h)
                    },         
    "stop_cost_acc": {'weight': 0.0,
                        'max_limit': 0.1}, # ? 
    "smooth": {'weight': 0.0}, # smoothness
    "state_bound": {'weight': 1000}, # joint limit avoidance
}
mppi_space_defaults = {'horizon': 30,
                        'n_iters': 1,
                        'num_particles': 500
                    }

