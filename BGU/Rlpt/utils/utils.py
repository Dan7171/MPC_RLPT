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
    if current < 0 or current > max_steps:
        raise ValueError("current must be between 0 and max_steps.")
    
    remaining_ratio = current / max_steps
    remaining_length = int(remaining_ratio * bar_length)
    bar = "[" + "=" * remaining_length + " " * (bar_length - remaining_length) + "]"
    color_print(f"\r{bar} {current}/{max_steps} steps. Seconds passed: {seconds_passed if seconds_passed != -1 else 'unavailable'}", end="")

