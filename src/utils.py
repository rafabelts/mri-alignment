"""Small utilities shared between modules"""

import torch


def get_device():
    """Solve using cuda if a GPU is available, otherwise, use cpu."""
    return "cuda" if torch.cuda.is_available() else "cpu"
