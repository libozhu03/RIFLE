"""Device selection and cache management for tiled VAE inference."""

import torch


def has_mps():
    return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()


def get_optimal_device_name():
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if has_mps() else "cpu"


def get_optimal_device():
    return torch.device(get_optimal_device_name())


device = get_optimal_device()


def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    if has_mps():
        torch.mps.empty_cache()
