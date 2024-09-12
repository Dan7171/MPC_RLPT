
## Disclaimer
This is a modified version of the [NVlabs/storm](https://github.com/NVlabs/storm) project.
This original project with all the modifications we performed in, can be found at: [./storm](./storm). 
- Modifications:
    - Adding the capability to tune Storm- Mpc's hyper-parameters rapidly using RL.
## License
This project inherits the MIT License of the original storm project. 
Link to liscene: [./storm/LICENSE](./storm/LICENSE). 
## Requirements:
- anaconda https://www.anaconda.com/download/success 
- python >= 3.8 https://www.python.org/downloads/
- linux (we used ubuntu >= 20.04)
- 

## Docs:
### Storm: 
- MPC_RPT/storm/docs/_build/html/index.html # ALT + l + ALT + o
- or STORM_DOCS/docs/_build/html/index.html # ALT + l + ALT + o
### Isaac Gym: 
- html: LINK MPC_RLPT/storm/isaacgym/docs/index.html # ALT + l + ALT + o
- pdfs: LINK MPC_RPT/GYM_DOCS 


## Installation 

- Only at first time you start working on your pc:

    - open a shell terminal
    - `cd /home/%user%/%path%/%to%/%project%/%parent_dir%` # replace with yours
    - `git clone <project url>` to clone the project into your parent_dir
    - `cd MPC_RLPT`to enter the git project
    - `conda create --name <env> --file ./environment.yml`  or `conda env create -f environment.yml` 

    creating a conda environment based on the environment file with all the requirements. Some modifications may be reaquired here! (Based on Hardware configuration such as GPUs for example which determines cuda...)
    - `export LD_LIBRARY_PATH=//home/%user%/anaconda3/envs/storm_kit/lib` // path to your anaconda3 dir - mine is /home/dan/anaconda3/envs/storm_kit/lib 
    - `cd '/storm`
    - `pip install -e .`
    - `cd '/storm/isaacgym/python` // I believe its necessary 1
    - `pip install -e .` // Read ./storm/isaacgym/README.txt
    - `conda env update --file environment.yml --prune`
    - `conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia`  

## Running Examples
- Every time you interact with system
    - `cd /home/%user%/%path%/%to%/%project%/%parent_dir%/MPC_RLPT`  
    - `conda activate --name <env>` // -> should now see (env) /home/%user%/%path%/%to%/%project%/%parent_dir%`
    - `export LD_LIBRARY_PATH=/home/dan/anaconda3/envs/storm_kit/lib`
    
    - Now you can run scripts!

        `python ./storm/examples/franka_reacher_for_rl.py`
        Example: (storm_kit) dan@dan-US-Desktop-Codex-R:~/sw/thesis/MPC_RLPT$ python ./storm/examples/franka_reacher.py 


 



# Reinforcement Learining Parameter Tuner (RLTP) for Storm Mpc



### What is Storm-Mpc?
#### Storm

The technical name is "Stochastic Tensor Optimization for Robot Motion". But this name is not really telling the whole picture. We suggest remembering what Storm is by the following attributs:

Storm is:

- An Integrated Framework for Fast Joint-Space Model-Predictive Control (Mpc) for Reactive Manipulation. [(website)](https://sites.google.com/view/manipulation-mpc/home)

- A GPU Robot Motion Toolkit, using NVIDIA's isaac-gym under the hood. Specifically its simulation manipulation at the moment. [(Storm README)](./storm/README.md)

- **[Most importantly to our project]** A framework which leverages **MPPI** to optimize over sampled actions and their costs. The costs are computed by rolling out the forward model from the current state with the sampled actions. [(Storm README)](./storm/README.md). 

**MPPI** is a variant and an implementation of an more general algorithmic pattern called a **"Sampling Based Mpc"** [(source)](https://markus-x-buchholz.medium.com/model-predictive-path-integral-mppi-control-in-c-b13ea594ca20) 

We'll now discuss what **Sampling Based Mpc**s are, as one of the core ideas which are required to understand Storm's project ans our work as a result. 
    
What is is a **Sampling Based Mpc**?
- First let's define the term **"Mpc"- Model Predictive Control**: 
    MPCs are a family of algorithms for controlling systems. Every MPC uses a mathematical **model** of the system being **controlled** to **predict** its future behavior. 

- **How Mpcs are working?** The main advantage of MPC is the fact that it allows the current timeslot to be optimized, while keeping future timeslots in account. This is achieved by optimizing a finite time-horizon, but only implementing the current timeslot and then optimizing again, repeatedly. [(wikipedia)](https://en.wikipedia.org/wiki/Model_predictive_control) 

Now that we know what Mpc is, what is a **Sampling Based Mpc** then?

a **Sampling-based MPC**  is a type of MPCs that doesn't rely on solving complex optimization problems directly. Instead, it uses a sampling approach to approximate the optimal solution. This means that on every timestamp, it generates a bunch (we call that bunch size the *number of particles*) of potential future trajectories (we call those trajectories also *rollouts* or *particles*) of the system, evaluates each of them seperately(using a *cost function*), and then selects the best one based on some criteria (the *cost function*). It's like looking at multiple possible paths into the future and choosing the one that seems the best. Each trajectory/rollout/path is like an "imaginary" simulation in the MPCs "mind". Also each "rollout" is calculated of a *finite horizon - H*, meaning how "deep" or "far" we let the mpc look into the future. After its generating rollouts and gets the new state and cost of, its updating the disturibution 
    
- **Pseudo code explained** *for personal use - will be removed in future.* To understand how the **sampling based mpc in Storm's paper**  (MPPI) is working, I deciphered the pseudo code from paper:
[link to psudo code of storm's mpc](https://docs.google.com/document/d/1CD7iObyP0k57gRCfo41qpGV2F-6hcC0CKYMkpfdAiJY/edit?usp=sharing)



So to wrap-up by **Storm-Mpc** we refer to the planning and optimization algorithm used by [Storm's paper](https://arxiv.org/pdf/2104.13542) under *Algorithm 1*, which its pseudo code is brought here: [link to psudo code of storm's mpc](https://docs.google.com/document/d/1CD7iObyP0k57gRCfo41qpGV2F-6hcC0CKYMkpfdAiJY/edit?usp=sharing). 

### What is RLPT?
RLPT is the name we give our Reinforcement Learning Parameter Tuner. It's one and only job is to rapidly (on every timestamp which planning in the Mpc begins) modify the parameters that are passed to the MPc, and help it generate more helpful/efficient trajacetories. Except starting at timestamp t with a new set of parameters everytime, there is no change in the MPc. More specificaly, Its true that RLPT switches parameters at every time unit t, but from the moment it switched it and the Mpc started its work, it will perform the exact same flow it used to in Storm, (1. K optimization steps of: 1.1 action sequences sampling 1.2 trajectory running 1.3 policy distribution update (optimizing) and 2. sample an action from the ptimized policy and passing it to the controller).   



# Project Roadmap

## Milestones

1.**Create RLPT infrastructure for static environment training** ![In Progress](https://img.shields.io/badge/-In%20Progress-yellow) 
   - Establish the infrastructure required for reinforcement learning parameter tuner (RLPT) in a static and fully observable environment.
   1. Storm environment (already static and fully observable) bring-up ![DONE](https://img.shields.io/badge/-DONE-green)
   2. Select the parameters to tune and their sample space.![In Progress](https://img.shields.io/badge/-In%20Progress-yellow).
   3. Develop a test environment to validate the parameter changing affect. ![TODO](https://img.shields.io/badge/-TODO-red) 
   4. Define the RLTP  training algorithm and implement it ![TODO](https://img.shields.io/badge/-TODO-red)  

2.**Define real-world problem cases which we want to test the improved system on** ![TODO](https://img.shields.io/badge/-TODO-red) 
- Prepering simulation environments for this use cases. ![TODO](https://img.shields.io/badge/-TODO-red) 
- Test new system comparing to standard Storm and show improvement. ![TODO](https://img.shields.io/badge/-TODO-red) 

3. **Add Moving Objects Support** ![TODO](https://img.shields.io/badge/-TODO-red) 
   - Enhance the RLPT infrastructure to include support for moving objects within the environment.
   


### Status Legend
- ![DONE](https://img.shields.io/badge/-DONE-green) - Completed tasks
- ![In Progress](https://img.shields.io/badge/-In%20Progress-yellow) - Tasks that are currently being worked on
- ![TODO](https://img.shields.io/badge/-TODO-red) - Tasks that are yet to be started


 