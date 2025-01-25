from operator import xor
from pyexpat import model
import subprocess
import os
import time

import torch
from BGU.Rlpt.configs.default_main import load_config_with_defaults
from BGU.Rlpt.utils.utils import color_print, make_model_path

if __name__ == '__main__':
    rlpt_cfg = load_config_with_defaults('BGU/Rlpt/configs/main.yml')
    agent_cfg = rlpt_cfg['agent']
    training = agent_cfg['training']['run']
    testing = agent_cfg['testing']['run']
    assert xor(training, testing), 'in external mode, training xor testing only'
    load_model_file = agent_cfg['model']['load_checkpoint']
    
    if load_model_file:
        checkpoint_path = agent_cfg['model']['checkpoint_path']
        assert os.path.exists(checkpoint_path), 'Model file does not exist'    
        model_file_path = checkpoint_path
        
        
    if training:    
        n_episodes = agent_cfg['training']['n_episodes'] # total for the whole training
        if load_model_file:
            ep_start = torch.load(model_file_path)['episode']  # episode to start from 
            color_print(f'using existing model: {model_file_path}')
            color_print(f'episode index was updated to {ep_start}')
            
        else:
            ep_start = 0
            model_file_path =  make_model_path(agent_cfg['model']['dst_dir'])
            print(f'new model {model_file_path} will be used')
    else: # testing    
        n_episodes = agent_cfg['testing']['n_episodes']
    
    # copy the config file of current run to the model dir
    
    script_name = 'franka_reacher_rlpt.py' if agent_cfg['alg'] == 'ddqn' else 'franka_reacher_rlpt_rainbow.py'
    for ep in range(ep_start, n_episodes):
        color_print(f'EXTERNAL RUNNER: episode: {ep} starts...', back_color='blue')        
        # Run the subprocess
        process = subprocess.Popen(
            # ['conda', 'run', '-n', 'storm_kit', 'python', '/home/dan/MPC_RLPT/BGU/Rlpt/franka_reacher_rlpt.py', '--external_run', 'True', '--model_path', model_file_path],
            ['conda', 'run','--no-capture-output', '-n', 'storm_kit', 'python','-u', f'/home/dan/MPC_RLPT/BGU/Rlpt/{script_name}', '--external_run', 'True', '--model_path', model_file_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr with stdout
            text=True,  
            bufsize=1 # Use text mode for easier handling of output
        )
        if process.stdout is not None:
            # Print each line of output as it arrives
            for line in process.stdout:
                print(line, end="")  # Print each line in real time
            # Wait for the subprocess to finish
            process.wait()
        