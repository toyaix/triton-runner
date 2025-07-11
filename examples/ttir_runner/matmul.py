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
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pid_n = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # initialize pointers for a and b
    a_ptrs = a_ptr + offs_m[:, None] * stride_am
    b_ptrs = b_ptr + offs_n[None, :] * stride_bn

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # accumulate along the N dimension
    for k in range(K):
        a_ptrs_iter = a_ptrs + k * stride_ak
        b_ptrs_iter = b_ptrs + k * stride_bk
        # load current blocks of a and b with boundary check
        a = tl.load(a_ptrs_iter)
        b = tl.load(b_ptrs_iter)
        accumulator += a * b

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
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))

    kernel_name = "matmul_kernel"
    ttir_path = os.path.join(current_dir, f"{kernel_name}.ttir")
    save_path = current_dir

    from triton_ml_runner.compile_utils import save_cubin_from_ttir
    save_cubin_from_ttir(ttir_path, kernel_name, save_path)

    metadata_path = os.path.join(save_path, f"{kernel_name}.json")
    cubin_path = os.path.join(save_path, f"{kernel_name}.cubin")

    from triton_ml_runner.cubin_utils import get_cufunction, cubin_launch
    function = get_cufunction(metadata_path, cubin_path, f"{kernel_name}")
    bound_args = (a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), 16,
                  16)
    signature_str = "* * * i32 i32 i32 i32 constexpr i32 constexpr i32 constexpr constexpr constexpr"
    grid = (
        triton.cdiv(N, 16),
        triton.cdiv(M, 16),
    )
    cubin_launch(function, signature_str, bound_args, grid)
    return c

    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(N, META['BLOCK_SIZE_N']),
        triton.cdiv(M, META['BLOCK_SIZE_M']),
    )
    matmul_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_N=16,
    )
    return c


torch.manual_seed(0)
a = torch.randn((512, 1024), device=DEVICE, dtype=torch.float32)
b = torch.randn((1024, 256), device=DEVICE, dtype=torch.float32)
torch_output = torch.matmul(a, b)
triton_output = matmul(a, b)

if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")
