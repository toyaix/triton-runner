def get_device_interface():
    import torch
    return torch.cuda


def get_empty_cache_for_benchmark():
    import torch

    # We maintain a buffer of 256 MB that we clear
    # before each kernel call to make sure that the L2 cache
    # doesn't contain any input data before the run
    cache_size = 256 * 1024 * 1024
    return torch.empty(int(cache_size // 4), dtype=torch.int, device='cuda')


def clear_cache(cache):
    cache.zero_()
