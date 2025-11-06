import logging
from datetime import datetime
import os
from Args import args
log_directory = "./log"


if not os.path.exists(log_directory):
    os.makedirs(log_directory)


log_subdirectory = os.path.join(log_directory, args.data)
if not os.path.exists(log_subdirectory):
    os.makedirs(log_subdirectory)


log_filename = os.path.join(
    log_subdirectory, f"{args.name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
)


logging.basicConfig(
    filename=log_filename,  
    level=logging.INFO,  
    format="%(asctime)s - %(levelname)s - %(message)s", 
    datefmt="%Y-%m-%d %H:%M:%S",  
)
def log_print(*args, **kwargs):
    message = " ".join(map(str, args))  
    logging.info(message)  
    print(message, **kwargs) 

