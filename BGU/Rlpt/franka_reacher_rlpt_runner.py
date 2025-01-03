import subprocess
import os
if __name__ == '__main__':
    conda_env_name = 'storm_kit'
    print(os.getcwd())
    subprocess.run(['conda', 'run', '-n', conda_env_name, 'python', '/BGU/Rlpt/franka_recher_rlpt.py'])