import logging
import os
import colorlog
from datetime import datetime
import numpy as  np
np.set_printoptions(precision=3)

logger_level = logging.INFO  # modifiable

initialized = False
fh_name = ''
logger = None
logger_ticks = 50 # log every  {logger_ticks} time stamps (arm steps)

handlers_dir_parent = './BGU/Rlpt/DebugLogs'
# # Check if the logger has handlers already attached to avoid duplicate logs
# if not logger.hasHandlers():
if not initialized:
    initialized = True

    if not os.path.exists(handlers_dir_parent):
        os.makedirs(handlers_dir_parent)
    current_time_string = datetime.now().strftime("%m-%d_%H-%M-%S")
    fh_name = current_time_string
    handler_path = f'{handlers_dir_parent}/{fh_name}.log'
    logger = logging.getLogger('my_project_logger')
    
    # Set the global logging level
    logger.setLevel(logger_level)

    # Create a console handler and set its logging level
    ch = logging.StreamHandler()
    ch.setLevel(logger_level)

    # Create a formatter and set it for the handler
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(funcName)s ::: %(message)s')
    # formatter = logging.Formatter('%(module)s-%(funcName)s:::%(message)s')

    # Define a formatter that adds color
    formatter_console = colorlog.ColoredFormatter(
        "%(log_color)s-%(module)s.py/%(funcName)s() : %(message)s",
        datefmt=None,
        reset=True,
        log_colors={
            'DEBUG':    'cyan',
            'INFO':     'green',
            'WARNING':  'yellow',
            'ERROR':    'red',
            'CRITICAL': 'bold_red',
        }
    )
    formatter_file =logging.Formatter('%(module)s.py/%(funcName)s() : %(message)s') 
    
    ch.setFormatter(formatter_console)
    
    # Add the handler to the logger
    logger.addHandler(ch)

    # Optionally, add a file handler if you want to log to a file as well
    fh = logging.FileHandler(handler_path)
    fh.setLevel(logger_level)
    fh.setFormatter(formatter_file)

    logger.addHandler(fh)
  
    