import torch
import numpy as np
import random

# Set the random seed for reproducibility
def set_seed(seed_value=42):

    random.seed(seed_value)       # Python random module
    np.random.seed(seed_value)    # Numpy module
    torch.manual_seed(seed_value) # PyTorch
    #env.seed(seed_value) # for older versions of gym
    #env.action_space.seed(seed_value)

    # if using PyTorch and you want deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)  # if using multi-GPU