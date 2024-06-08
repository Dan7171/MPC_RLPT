class Rlpt:

    """Reinforcement Learning Paramer Tuner for an MPC paremeters (currently for STORMs MPC only)"""
    
    def __init__(self,episode_num:int, episode_max_step:int, Mpc):
        self.episode_num = episode_num # how many episodes to run in total
        self.episode_max_step = episode_max_step # duration in time units (number of MPC steps) on each episde
        self.mpc = Mpc()
        
    def train(self, episode_num:int):
        """
        Training the Rlpt  
        """
        for ep_index in range(self.episode_num):
            self.run_episode(self.episode_max_step) # runs an entire episode of the simulator using the MPc
            self.Mpc.reset() # reset the environment (change location of obstacles and target object)
    
    def get_action_space():
        # todo
        pass


    def run_episode(self, max_step:int):
        """
        max_step: the maximal number of steps (maximum duration/time) for an episode. If reached, episode is over even target location was not reached.
        """
        
        dict: cost_params;
        dict: mpc_params;
        
        for step_index in range(max_step):
            
            # Select best action for the tuner - an assignment for the MPC parameters
            cost_params, mpc_params = RLPT.select_action(state) # this is not yet implemented, that's what our RL parameter tuner should do
            
            a = (cost_params, mpc_params) # select RLTP action

            # Perform the mpc-step with the "tuned" parameters  
            mpc_step_outputs_all = Mpc.step(a)
            
            # original Mpc.step() output
            arm_configuration, goal_pose, end_effector_pos, end_effector_quat, done = mpc_step_outputs_all[:-1]
            # other outputs we added to be used by RLPT
            mpc_step_outputs_for_rlpt = mpc_step_outputs_all[-1] # new output. Here we put anything which essential for the RLTP to update its policy
            
            # Given the essential information from MPC.step()'s output, update policy_update policy of RLPT
            s, r, s_next = mpc_step_outputs_for_rlpt # this brought as classic RL but can be replaced with 
            RLTP.policy_update(mpc_step_outputs_for_rlpt)
            
            if done: # episode was eneded before reaching the max episode step, which is good! (arm reached the target)
                break # stop making steps of this episode

        

            