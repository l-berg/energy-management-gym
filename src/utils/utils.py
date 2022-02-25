
import torch
import numpy as np
import random


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def get_device():
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


