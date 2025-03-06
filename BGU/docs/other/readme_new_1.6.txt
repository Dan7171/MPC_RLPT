# start working after turning on pc #:


cd /home/dan/storm_dan/storm/examples
conda activate storm_kit
# export LD_LIBRARY_PATH=/home/dan/anaconda3/envs/storm_kit/lib NOTE: this is crucial for every bash terminal, unless you update anaconda3/envs/storm_kit/etc/conda/activate.d/env_vars.sh and deactivte.d/env_vars.sh files with export and unset as I did
python franka_reacher_for_rl.py # can also run franka_reacher.py, franka_reacher_for_rl_comparison.py 







# USEFUL #

1. get python version:
(storm_kit) dan@dan-US-Desktop-Codex-R:~/sw/thesis/storm_dan/storm/examples$ python --version
Python 3.8.19

2. Using austin to profile times:
from conda storm_kit env:

cd /home/dan/storm_dan
Run one of those:

1. Standard command austin -o ./f.austin python ./storm/examples/franka_reacher.py  # 100 micro seconds interval between sampling
2. OR: austin -o ./f.austin  -i 5000 python ./storm/examples/franka_reacher_for_rl_comparison.py  # the -i decreses sampling rate and makes smaller austin files - 5000 micro seconds
3. OR: austin -o ./f.austin -i 1s python "./storm/examples/franka_reacher_for_rl_comparison_dan.py" # 1 second interval between follwing samples

run...
ctrl+c to kill
ctrl+shift+p load austin samples
select the ./f.austin file and see flame graph in terminal
