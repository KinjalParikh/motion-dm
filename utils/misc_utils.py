import torch
import random
import numpy as np


def fixseed(seed):
    torch.backends.mps.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    # if torch.cuda.is_available():
    #     return "cuda"
    # elif torch.backends.mps.is_available():
    #     return "mps"
    # else:
    return "cpu"
