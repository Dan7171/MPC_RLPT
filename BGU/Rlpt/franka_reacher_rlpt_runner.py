from json import load
import subprocess
import os
from BGU.Rlpt.configs.default_main import load_config_with_defaults
from BGU.Rlpt.utils.utils import make_model_path
if __name__ == '__main__':
    rlpt_cfg = load_config_with_defaults('BGU/Rlpt/configs/main.yml')['agent']['training']
    episodes = rlpt_cfg['training']['n_episodes'] # total for the whole training
    load_model_file = rlpt_cfg['model']['load_checkpoint']    
    model_file_exists = load_model_file
    model_file_path = rlpt_cfg['model']['checkpoint_path'] if load_model_file else make_model_path(rlpt_cfg['model']['models_dst_dir']) 
    
    for e in range(episodes):
        subprocess.run(['conda', 'run', '-n', 'storm_kit', 'python', '/home/dan/MPC_RLPT/BGU/Rlpt/franka_reacher_rlpt.py', '--external_run', 'True'])
        model_file_exists = True