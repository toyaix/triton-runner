import torch
import triton

import triton.language as tl

import triton_runner


@triton.jit
def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck,
                  BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid_k = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an
    b_ptrs = b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        max_idx = N - n * BLOCK_SIZE_N
        a = tl.load(a_ptrs + n * BLOCK_SIZE_N * stride_an, mask=offs_n[None, :] < max_idx, other=0.0)
        b = tl.load(b_ptrs + n * BLOCK_SIZE_N * stride_bn, mask=offs_n[:, None] < max_idx, other=0.0)
        accumulator = tl.dot(a, b, acc=accumulator)
        # tl.dot has bug when sm < 80, comment out the code above and enable the code below.
        # accumulator = tl.dot(a, b, acc=accumulator, input_precision="ieee")

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_k[None, :] * stride_ck
    c_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    tl.store(c_ptrs, accumulator, mask=c_mask)


@triton_runner.jit
def runner_matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck,
                         BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid_k = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)

    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an
    b_ptrs = b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
        max_idx = N - n * BLOCK_SIZE_N
        a = tl.load(a_ptrs + n * BLOCK_SIZE_N * stride_an, mask=offs_n[None, :] < max_idx, other=0.0)
        b = tl.load(b_ptrs + n * BLOCK_SIZE_N * stride_bn, mask=offs_n[:, None] < max_idx, other=0.0)
        accumulator = tl.dot(a, b, acc=accumulator)
        # tl.dot has bug when sm < 80, comment out the code above and enable the code below.
        # accumulator = tl.dot(a, b, acc=accumulator, input_precision="ieee")

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_k[None, :] * stride_ck
    c_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    tl.store(c_ptrs, accumulator, mask=c_mask)
