if __name__ == '__main__':
    # instantiate empty gym:
    parser = argparse.ArgumentParser(description='pass args')
    parser.add_argument('--robot', type=str, default='franka', help='Robot to spawn')
    parser.add_argument('--cuda', action='store_true', default=True, help='use cuda')
    parser.add_argument('--headless', action='store_true', default=False, help='headless gym')
    parser.add_argument('--control_space', type=str, default='acc', help='Robot to spawn')
    args = parser.parse_args()
    
    sim_params = load_yaml(join_path(get_gym_configs_path(),'physx.yml'))
    sim_params['headless'] = args.headless
    gym_instance = Gym(**sim_params)
    
    
    Mpc = MpcRobotInteractive(args, gym_instance)
    
    i = 0
    end_flag = True
    steps_in_one_epispde = 1000 # dan - this is the number of time units I guess which and episode takes
    while(i > -100):
        # >>> Dan 
        # Dan - to play with the non 0 cost params and see the affect.
        # <<<
        cost_params = {
            "manipulability": 500, # 30 
            "stop_cost": 50, 
            "stop_cost_acc": 0.0, 
            "smooth": 0.0, 
            "state_bound": 1000.0, 
            "ee_vel": 0.0, 
            "robot_self_collision" : 5000, 
            "primitive_collision" : 5000, 
            "voxel_collision" : 0
            }
        mpc_params = {
            "horizon" : 90 , # Dan - From paper:  How deep into the future each rollout (imaginary simulation) sees
            "particles" : 500 # Dan - How many rollouts are done. from paper:Number of trajectories sampled per iteration of optimization (or particles)
            } #dan
        arm_configuration, goal_pose, end_effector_pos, end_effector_quat, done = Mpc.step(cost_params, mpc_params, i)
        if end_flag:
            start_time = time.time()
            end_flag = False
        end_time = 0
        if i%steps_in_one_epispde == 0: 
            end_time = time.time()
            end_flag = True
            elapsed_time = end_time - start_time
            print(f"Execution time: {elapsed_time:.6f} seconds!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
            Mpc.reset()
        if done:
            break
        i += 1
    