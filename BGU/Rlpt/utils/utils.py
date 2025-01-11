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

def color_print(text, fore_color='black', back_color='green'):
    fore = {'red':Fore.RED, 'green':Fore.GREEN,'black':Fore.BLACK, 'blue':Fore.BLUE}
    back = {'red':Back.RED, 'green':Back.GREEN,'black':Back.BLACK, 'blue':Back.BLUE}
    
    print(fore[fore_color] + '')
    print(back[back_color] + text)    
    print(Style.RESET_ALL)



