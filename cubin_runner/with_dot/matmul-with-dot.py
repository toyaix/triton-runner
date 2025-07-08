import triton
import triton.language as tl
import torch

DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_n = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # initialize pointers for a and b
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # accumulate along the N dimension
    for k in range(tl.cdiv(K, BLOCK_SIZE_K)):
        max_idx = K - k * BLOCK_SIZE_K
        # load current blocks of a and b with boundary check
        a = tl.load(a_ptrs + k * BLOCK_SIZE_K * stride_ak, mask=offs_k[None, :] < max_idx, other=0.0)
        b = tl.load(b_ptrs + k * BLOCK_SIZE_K * stride_bk, mask=offs_k[:, None] < max_idx, other=0.0)
        # cal a @ b to accumulator
        accumulator = tl.dot(a, b, acc=accumulator)

    # write result back to c
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def matmul(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)

    import sys
    sys.path.append('..')

    from utils import get_cufunction, cubin_launch
    kernel_name = "matmul_kernel"
    function = get_cufunction(f"{kernel_name}.json", f"{kernel_name}.cubin", f"{kernel_name}")
    bound_args = (a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1),
                  c.stride(0), c.stride(1), 128, 64, 64)
    signature_str = "* * * i32 i32 i32 i32 constexpr i32 constexpr i32 constexpr constexpr constexpr constexpr"
    grid = (triton.cdiv(N, 64), triton.cdiv(M, 128), )
    cubin_launch(function, signature_str, bound_args, grid)

    # 1D launch kernel where each block gets its own program.
    # grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE_N']), triton.cdiv(M, META['BLOCK_SIZE_M']), )
    # matmul_kernel[grid](
    #     a, b, c,
    #     M, N, K,
    #     a.stride(0), a.stride(1),
    #     b.stride(0), b.stride(1),
    #     c.stride(0), c.stride(1),
    #     BLOCK_SIZE_M=128,
    #     BLOCK_SIZE_N=64,
    #     BLOCK_SIZE_K=64,
    # )
    return c


torch.manual_seed(0)
a = torch.randn((512, 1024), device=DEVICE, dtype=torch.float16)
b = torch.randn((1024, 256), device=DEVICE, dtype=torch.float16)
torch_output = torch.matmul(a, b)
triton_output = matmul(a, b)

if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")
