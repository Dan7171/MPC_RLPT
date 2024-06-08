franka_reacher_for_rl_comparison.py:
    1. to select a world non- deterministically:
         comment out:  #world_params, indexes, compressed_world_params = self.modify_dict(world_yml) # modify dict - randonmly seclecting a world
         and comment  "iter" (iter is the index of the world to select deterministically)
    2. to select a world deterministically: the exact oposite (comment out iter and comment modify_dict)


# the "comparison" script's goal is to see the difference in arm motion under different parameter setting BUT in the same world. 

# world_params:= all the objects that are in the simulator
# compressed_world_params:= only the subset of world_params which is ON the table, and not "outside" (inside the cube). Its done by elias to make less computation on the objects that are outside to the taeble, and sending only the compressed_world_params to the code of STORM that making all the computation.


# franka_reacher_for_rl_comparison.py - just for "debbuging" and comparring behaviours
# franka_reacher_for_rl.py - the most important file, the one we finally use

# ACTION ITEMS:
1. to set a meeting with elias and go over the files
