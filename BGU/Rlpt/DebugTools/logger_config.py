import logging
import time
import os
# Create a logger object

initialized = False
fh_name = ''
logger = None
handlers_dir_parent = './BGU/Rlpt/DebugLogs'
# # Check if the logger has handlers already attached to avoid duplicate logs
# if not logger.hasHandlers():
if not initialized:
    initialized = True

    if not os.path.exists(handlers_dir_parent):
        os.makedirs(handlers_dir_parent)
    fh_name = str(time.time())
    handler_path = f'{handlers_dir_parent}/{fh_name}.log'
    logger = logging.getLogger('my_project_logger')
    
    # Set the global logging level
    logger.setLevel(logging.DEBUG)

    # Create a console handler and set its logging level
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # Create a formatter and set it for the handler
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(module)s - %(funcName)s ::: %(message)s')
    ch.setFormatter(formatter)

    # Add the handler to the logger
    logger.addHandler(ch)

    # Optionally, add a file handler if you want to log to a file as well
    fh = logging.FileHandler(handler_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
  
    