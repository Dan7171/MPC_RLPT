from operator import xor
from pyexpat import model
import subprocess
import os
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
        if not load_model_file:
            model_file_path =  make_model_path(agent_cfg['model']['models_dst_dir'])

    else: # testing    
        n_episodes = agent_cfg['testing']['n_episodes']
    
    for e in range(n_episodes):
        subprocess.run(['conda', 'run', '-n', 'storm_kit', 'python', '/home/dan/MPC_RLPT/BGU/Rlpt/franka_reacher_rlpt.py', '--external_run', 'True', '--model_path', model_file_path])
        