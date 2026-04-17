import triton
import triton.language as tl
import torch
import triton_runner

triton_runner.configure_jit_backend()

DEVICE = triton_runner.torch_utils.get_active_torch_device()


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(axis=1)
    pid_n = tl.program_id(axis=0)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        accumulator = tl.dot(a, b, acc=accumulator)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)


def matmul(a, b, source_type=None, source_text=None, metadata_json=None):
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE_N']), triton.cdiv(M, META['BLOCK_SIZE_M']), )
    kwargs = dict(
        BLOCK_SIZE_M=16, BLOCK_SIZE_N=16, BLOCK_SIZE_K=16,
    )
    if source_type is not None:
        kwargs[source_type] = source_text
    if metadata_json is not None:
        kwargs["metadata_json"] = metadata_json
    compiled = matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        **kwargs,
    )
    return c, compiled


torch.manual_seed(0)
a = torch.randn((512, 1024), device=DEVICE, dtype=torch.float32)
b = torch.randn((1024, 256), device=DEVICE, dtype=torch.float32)
torch_output = torch.matmul(a, b)

# First run: normal compile, extract the generated PTX text.
triton_output, compiled = matmul(a, b)
matmul_ttgir_src = compiled.asm["ttgir"]
matmul_llir_src = compiled.asm["llir"]
matmul_ptx_src = compiled.asm["ptx"]
matmul_metadata_json = compiled.metadata

# Second run: feed the generated lowerings back via source strings.
triton_output_from_ttgir, _ = matmul(a, b, source_type="ttgir_src", source_text=matmul_ttgir_src)
triton_output_from_llir, _ = matmul(a, b, source_type="llir_src", source_text=matmul_llir_src, metadata_json=matmul_metadata_json)
triton_output_from_ptx, _ = matmul(a, b, source_type="ptx_src", source_text=matmul_ptx_src, metadata_json=matmul_metadata_json)

if (torch.allclose(triton_output, torch_output, atol=1e-1, rtol=1e-2)
        and torch.allclose(triton_output_from_ttgir, torch_output, atol=1e-1, rtol=1e-2)
        and torch.allclose(triton_output_from_llir, torch_output, atol=1e-1, rtol=1e-2)
        and torch.allclose(triton_output_from_ptx, torch_output, atol=1e-1, rtol=1e-2)):
    print("✅ Triton and Torch match")
else:
    print(abs(triton_output - torch_output).max())
    print(abs(triton_output_from_ttgir - torch_output).max())
    print(abs(triton_output_from_llir - torch_output).max())
    print(abs(triton_output_from_ptx - torch_output).max())
    print("❌ Triton and Torch differ")
