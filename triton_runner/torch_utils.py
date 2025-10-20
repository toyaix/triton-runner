import torch
import math
import triton


def pad_2d_to_block_shape(tensor, block_shape):
    M, K = tensor.shape
    BLOCK_M, BLOCK_K = block_shape

    pad_M = (BLOCK_M - M % BLOCK_M) % BLOCK_M
    pad_K = (BLOCK_K - K % BLOCK_K) % BLOCK_K

    padded = torch.nn.functional.pad(tensor, (0, pad_K, 0, pad_M), value=0)

    return padded.to(torch.float32)


def get_pad_n_elements(tensor, block_shape):
    return math.prod(tuple(triton.cdiv(dim, block) * block for dim, block in zip(tensor.shape, block_shape)))


def get_grid_dim(tensor_shape, block_shape):
    return tuple(triton.cdiv(dim, block) for dim, block in zip(tensor_shape, block_shape))


def get_n_elements_with_grid(block_shape, grid):
    return math.prod(block_shape) * math.prod(grid)
