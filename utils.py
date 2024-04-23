import os
import random
import numpy as np
import torch
from rich.logging import RichHandler
import logging
try:
    import pretty_errors
except ImportError:
    print("pretty_errors is not installed. Error messages will not be colored.")
    print("Please install the 'pretty_errors' package using 'pip install pretty_errors'.")
    
def seed_everything(seed=3407):
    """
    Sets the seed for Python's random module, NumPy, and PyTorch to ensure reproducibility.

    Parameters:
    - seed: An integer used as the seed for random number generators. Default is 3407.

    Raises:
    - ImportError: If torch is not installed.
    """
    try:
        random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = True
        else:
            print("CUDA is not available. Seeding only for CPU.")
    except ImportError:
        print("torch is not installed. Seeding only for Python's random module and NumPy.")
        
        
def get_logger(name):
    """
    Set up and use a logger with RichHandler for pretty logging.
    This function handles exceptions that might occur during the setup or usage of the logger.
    """
    try:
        # Import statements are moved inside the function to encapsulate them
        from rich.logging import RichHandler
        import logging
    except ImportError as e:
        # Print the error message if the imports fail
        print(f"An error occurred: {e}")
        # remind the user to install the required package
        print("Please install the 'rich' package using 'pip install rich'")

    # Configure the basic logging settings    
    logging.basicConfig(
        level="DEBUG",
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler()]
    )

    # Get the logger with the specified name
    logger = logging.getLogger(name)

    # Use the logger with different levels
    logger.debug("Debug message")
    logger.info("Information message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")

    print(f"Logger in {name}.py setup and usage completed successfully.")
    return logger


def check_gpu_status(cfg):
    """
    Checks the GPU status and sets the environment variable for CUDA_VISIBLE_DEVICES if necessary.

    Parameters:
    - cfg: A configuration object that contains 'single_gpu' and 'device_id' attributes.

    Returns:
    - is_gpu: A boolean indicating if a GPU is available.
    - is_multi_gpu: A boolean indicating if multiple GPUs are available.
    """
    is_gpu, is_multi_gpu = False, False

    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        if device_count > 1:
            print(f"Number of GPUs: {device_count}")
            is_multi_gpu = True
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        is_gpu = True
    else:
        print("GPU is not available")

    if cfg.single_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.device_ids[0]
        print(f"Using GPU {cfg.device_ids[0]}")
        is_multi_gpu = False

    return is_gpu, is_multi_gpu