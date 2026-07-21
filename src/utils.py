"""Utilidades pequeñas compartidas entre módulos."""

import torch


def get_device():
    """Resuelve 'cuda' si hay GPU disponible, si no 'cpu'."""
    return "cuda" if torch.cuda.is_available() else "cpu"
