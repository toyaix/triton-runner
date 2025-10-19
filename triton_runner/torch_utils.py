import torch


def pad_2d_to_block_shape(tensor, block_shape):
    M, K = tensor.shape
    BLOCK_M, BLOCK_K = block_shape

    pad_M = (BLOCK_M - M % BLOCK_M) % BLOCK_M
    pad_K = (BLOCK_K - K % BLOCK_K) % BLOCK_K

    padded = torch.nn.functional.pad(tensor, (0, pad_K, 0, pad_M), value=0)

    return padded.to(torch.float32)
