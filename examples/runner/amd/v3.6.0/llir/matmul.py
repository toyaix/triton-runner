import triton
import triton.language as tl
import torch


DEVICE = triton.runtime.driver.active.get_active_torch_device()


import triton_runner


# @triton.jit
@triton_runner.jit
def mutmul_native(
    a_ptr,
    b_ptr,
    c_ptr,  # pointers to matrices A, B, and C in GPU memory
    M,
    N,
    K,  # dimensions: A[M,K], B[K,N], C[M,N]
    stride_am,
    stride_ak,  # strides for A: row stride (am), column stride (ak)
    stride_bk,
    stride_bn,  # strides for B: row stride (bk), column stride (bn)
    stride_cm,
    stride_cn,  # strides for C: row stride (cm), column stride (cn)
    BLOCK_SIZE_M: tl.constexpr,  # number of rows in a block of A and C
    BLOCK_SIZE_N: tl.constexpr,  # number of columns in a block of B and C
    BLOCK_SIZE_K: tl.constexpr,  # number of rows in a block of B and columns in A
):

    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    pid_n = pid % num_pid_n
    pid_m = pid // num_pid_n
    # Compute the row and column indices for this block
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)  # row indices
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)  # column indices
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    # Initialize the accumulator for C with zeros
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Loop over the shared dimension N
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)

        accumulator = tl.dot(a, b, accumulator)

        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c = accumulator.to(tl.float16)
    # Compute the pointers for the output block in C
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn

    # Create a mask to avoid writing out-of-bounds elements
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)

    # Store the computed block back into C
    tl.store(c_ptrs, c, mask=c_mask)
    pass


def matmul(a, b):
    M, K = a.shape
    K, N = b.shape
    stride_am, stride_ak = K, 1
    stride_bk, stride_bn = N, 1
    stride_cm, stride_cn = N, 1

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    # TMA descriptors require a global memory allocation
    def alloc_fn(size, alignment, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )

    mutmul_native[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=64,
        BLOCK_SIZE_K=64,
        llir_dir=triton_runner.get_file_dir(__file__),
    )

    return c


M, N, K = 1024, 512, 256

device = torch.cuda.current_device()

a = torch.rand((M, N), device=DEVICE, dtype=torch.float16) - 0.5
b = torch.rand((N, K), device=DEVICE, dtype=torch.float16) - 0.5

triton_output = matmul(a, b)
torch_output = torch.matmul(a, b)
print(f"triton_output={triton_output}")
print(f"torch_output={torch_output}")
if torch.allclose(triton_output.float(), torch_output.float(), atol=1e-3, rtol=1e-3):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")
