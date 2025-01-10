from operator import xor
from pyexpat import model
import subprocess
import os

import torch
from BGU.Rlpt.configs.default_main import load_config_with_defaults
from BGU.Rlpt.utils.utils import make_model_path

if __name__ == '__main__':
    agent_cfg = load_config_with_defaults('BGU/Rlpt/configs/main.yml')['agent']
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
            print(f'using existing model: {model_file_path}')
            print(f'ep was updated to {ep_start}')
            
        else:
            ep_start = 0
            model_file_path =  make_model_path(agent_cfg['model']['dst_dir'])
            print(f'new model {model_file_path} will be used')
    else: # testing    
        n_episodes = agent_cfg['testing']['n_episodes']
    
    # copy the config file of current run to the model dir


    for ep in range(ep_start, n_episodes):
        print(f'episode: {ep} starts...')
        subprocess.run(['conda', 'run', '-n', 'storm_kit', 'python', '/home/dan/MPC_RLPT/BGU/Rlpt/franka_reacher_rlpt.py', '--external_run', 'True', '--model_path', model_file_path])
        pass