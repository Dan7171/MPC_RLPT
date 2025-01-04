from pyexpat import model
import subprocess
import os
from BGU.Rlpt.configs.default_main import load_config_with_defaults
from BGU.Rlpt.utils.utils import make_model_path

if __name__ == '__main__':
    rlpt_cfg = load_config_with_defaults('BGU/Rlpt/configs/main.yml')['agent']['training']
    episodes = rlpt_cfg['training']['n_episodes'] # total for the whole training
    if load_model_file := rlpt_cfg['model']['load_checkpoint']:
        checkpoint_path = rlpt_cfg['model']['checkpoint_path']
        assert os.path.exists(checkpoint_path), 'Model file does not exist'    
        model_file_path = checkpoint_path
    else:
        model_file_path =  make_model_path(rlpt_cfg['model']['models_dst_dir'])
        
    for e in range(episodes):
        subprocess.run(['conda', 'run', '-n', 'storm_kit', 'python', '/home/dan/MPC_RLPT/BGU/Rlpt/franka_reacher_rlpt.py', '--external_run', 'True', '--model_path', model_file_path])
        model_file_exists = True