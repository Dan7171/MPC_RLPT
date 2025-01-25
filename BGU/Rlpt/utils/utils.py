import os
import time
from colorama import Fore, Back, Style
import psutil

def make_model_path(models_dst_dir):
    training_starttime = time.strftime('%Y:%m:%d(%a)%H:%M:%S')
    model_file_path = os.path.join(models_dst_dir, training_starttime, 'model.pth')  
    return model_file_path
 

def kill_zombie_processes():
    parent = psutil.Process(os.getpid())
    for child in parent.children(recursive=True):
        if child.status() == psutil.STATUS_ZOMBIE:
            child.kill()

def color_print(text, fore_color='black', back_color='green', end='\n'):
    fore = {'red':Fore.RED, 'green':Fore.GREEN,'black':Fore.BLACK, 'blue':Fore.BLUE}
    back = {'red':Back.RED, 'green':Back.GREEN,'black':Back.BLACK, 'blue':Back.BLUE}
    print(fore[fore_color] + '',end=end)
    print(back[back_color] + text,end=end)    
    print(Style.RESET_ALL)



def print_progress_bar(current, max_steps, bar_length=50,seconds_passed=-1.0):
    """
    Prints a progress bar to indicate remaining units of a variable.
    
    Parameters:
        current (int): The current step (remaining units).
        max_steps (int): The maximum number of steps.
        bar_length (int): The length of the bar in characters (default: 50).
    """
    if max_steps <= 0:
        raise ValueError("max_steps must be a positive integer.")
    if current <= 0 or current > max_steps:
        raise ValueError(f"current must be between 1 and max_steps. but ,{current} was given and max_steps = {max_steps}")
    
    remaining_ratio = current / max_steps
    remaining_length = int(remaining_ratio * bar_length)
    bar = "[" + "=" * remaining_length + " " * (bar_length - remaining_length) + "]"
    
    print(f"\r{bar} {current}/{max_steps} steps completed. Time: {seconds_passed if seconds_passed != -1 else '?'} seconds", end="")
    




def goal_test(pos_error, rot_error, goal_test_cfg) -> bool:
    """ performing goal test based on end effector position and rotation errors.

    Args:
        pos_error (_type_): _description_
        rot_error (_type_): _description_

    Returns:
        bool: _description_
    """
    pos_threshold = goal_test_cfg['goal_pos_thresh_dist']
    rot_threshold = goal_test_cfg['goal_rot_thresh_dist']
    requirements = goal_test_cfg['requirements']
    passed_pos_test = 'pos' not in requirements or pos_error < pos_threshold
    passed_rot_test = 'rot' not in requirements or rot_error < rot_threshold
    return passed_pos_test and passed_rot_test   