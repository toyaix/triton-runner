# The use of PyTorch in Triton programs is not allowed for the purposes of fair benchmarking.
import triton
import triton.language as tl

@triton.jit
def matmul_kernel_with_block_kernel(a_ptr, b_ptr, c_ptr,  #
                                        M, N, K,  #
                                        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                                        BLOCK_SIZE_K: tl.constexpr,  #
                                        ):
    a_ptr = a_ptr.to(tl.pointer_type(tl.float32))
    b_ptr = b_ptr.to(tl.pointer_type(tl.float32))
    c_ptr = c_ptr.to(tl.pointer_type(tl.float32))
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    a_block_ptr = tl.make_block_ptr(base=a_ptr, shape=(M, N), strides=(N, 1),
                                    offsets=(pid_m * BLOCK_SIZE_M, 0),
                                    block_shape=(BLOCK_SIZE_M, BLOCK_SIZE_N), order=(1, 0))
    b_block_ptr = tl.make_block_ptr(base=b_ptr, shape=(N, K), strides=(K, 1),
                                    offsets=(0, pid_k * BLOCK_SIZE_K),
                                    block_shape=(BLOCK_SIZE_N, BLOCK_SIZE_K), order=(1, 0))

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
    for n in range(tl.cdiv(N, BLOCK_SIZE_N)):
        a = tl.load(a_block_ptr, boundary_check=(1, ), padding_option="zero")
        b = tl.load(b_block_ptr, boundary_check=(0, ), padding_option="zero")
        accumulator = tl.dot(a, b, acc=accumulator, input_precision="ieee")
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_SIZE_N))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_SIZE_N, 0))

    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_ck = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    c_ptrs = c_ptr + offs_cm[:, None] * K + offs_ck[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_ck[None, :] < K)
    tl.store(c_ptrs, accumulator, mask=c_mask)


# a_ptr, b_ptr, c_ptr are raw device pointers
def solve(a_ptr: int, b_ptr: int, c_ptr: int, M: int, N: int, K: int):
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(K, META['BLOCK_SIZE_K']), )
    matmul_kernel_with_block_kernel[grid](
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_K=64,
        BLOCK_SIZE_N=64,
    )

M, N, K = 8192, 6144, 4096

import torch
device = torch.cuda.current_device()

A = torch.randn(M, N, device=device)
B = torch.randn(N, K, device=device)
C = torch.randn(M, K, device=device)
solve(A[None], B[None], C[None], M, N, K)
torch_output = torch.matmul(A, B)

if torch.allclose(C, torch_output, atol=0.125, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")
