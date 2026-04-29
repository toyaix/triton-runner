# Triton 3.3 Performance Regression on Small Gemms — BenchmarkKernel contract
# https://github.com/triton-lang/triton/issues/7096
#
# Adapted from test.py to the cross-version BenchmarkKernel interface.
# Uses hardcoded config: BLOCK_M=64, BLOCK_N=128, BLOCK_K=64, GROUP_M=8, 4 warps, 4 stages
# (matches the single autotune config from the original matmul.py)

import torch
import triton
from triton import language as tl
from triton_runner.bench.cross_version import ProblemSpace


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    c = accumulator.to(tl.float16)

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# ---- BenchmarkKernel contract ----

def prepare_args(M, N, K):
    """Generate input tensors for the given problem size."""
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    c = torch.empty((M, N), device="cuda", dtype=torch.float16)
    return (a, b, c, M, N, K,
            a.stride(0), a.stride(1),
            b.stride(0), b.stride(1),
            c.stride(0), c.stride(1))


def get_grid(M, N, K):
    """Compute launch grid for the hardcoded tile config (BLOCK_M=64, BLOCK_N=128)."""
    return (triton.cdiv(M, 64) * triton.cdiv(N, 128), 1, 1)


def get_kernel_kwargs(M, N, K):
    """Return constexpr + launch kwargs for the hardcoded tile config."""
    return dict(
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64,
        GROUP_SIZE_M=8, num_stages=4, num_warps=4,
    )


def get_problem_space():
    """Problem sizes from the original issue #7096 benchmark."""
    return ProblemSpace.matmul_square([512, 1024, 1536, 2048, 4096])


kernel = matmul_kernel
