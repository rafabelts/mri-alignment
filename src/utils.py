"""Small utilities shared between modules"""
import random
import numpy as np
import torch


def get_device():
    """Solve using cuda if a GPU is available, otherwise, use cpu."""
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(seed=42, deterministic=True):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
