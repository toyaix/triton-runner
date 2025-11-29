import torch
import triton
import triton.language as tl
import triton_runner

@triton_runner.jit
def matrix_multiplication_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_an,
    stride_bn, stride_bk,
    stride_cm, stride_ck,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_k = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    # initialize pointers for a and b
    a_ptrs = a_ptr + offs_m[:, None] * stride_am
    b_ptrs = b_ptr + offs_k[None, :] * stride_bk

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    # accumulate along the N dimension
    for n in range(N):
        # load current blocks of a and b with boundary check
        a = tl.load(a_ptrs + stride_an * n)
        b = tl.load(b_ptrs + stride_bn * n)
        accumulator += a * b

    # write result back to c
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_k[None, :] * stride_ck
    c_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    tl.store(c_ptrs, accumulator.to(c_ptr.dtype.element_ty), mask=c_mask)


def solve(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor, M: int, N: int, K: int):
    grid = lambda META: (triton.cdiv(K, META['BLOCK_SIZE_K']), triton.cdiv(M, META['BLOCK_SIZE_M']), )

    BLOCK_SIZE_M, BLOCK_SIZE_K = 64, 32
    dump_tensor = torch.empty((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=torch.float32, device=a.device)
    # dump_value can be "%accumulator_35"(acc in loop)
    dump_value = "%accumulator_35"
    dump_grid = (triton.cdiv(K, BLOCK_SIZE_K) - 1, triton.cdiv(M, BLOCK_SIZE_M) - 1)

    matrix_multiplication_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        ttir_dir=triton_runner.get_file_dir(__file__),
        dump_tensor=dump_tensor,
        dump_value=dump_value,
        dump_grid=dump_grid,
    )
    triton_runner.color_print.blue_print(f"debug {dump_tensor}")
    dump_torch = a @ b
    boundary_start_M = M & (~(BLOCK_SIZE_M-1))
    boundary_start_K = K & (~(BLOCK_SIZE_K-1))
    # k % BLOCK_SIZE_K == 0
    if K & (BLOCK_SIZE_K - 1) == 0:
        boundary_start_K -= BLOCK_SIZE_K
    max_diff = torch.max(torch.abs(dump_torch[boundary_start_M:, boundary_start_K:] - dump_tensor[:M-boundary_start_M, :K-boundary_start_K]))
    triton_runner.color_print.yellow_print(f"The maximum difference between torch and dump is {max_diff}")


if __name__ == "__main__":
    M, N, K = 210, 256, 192
    a = torch.randn((M, N), device='cuda', dtype=torch.bfloat16)
    b = torch.randn((N, K), device='cuda', dtype=torch.bfloat16)
    torch_output = a @ b
    triton_output = torch.empty(torch_output.shape, device='cuda', dtype=torch.bfloat16)
    solve(a, b, triton_output, M, N, K)
    if torch.allclose(triton_output, torch_output, atol=1e-01, rtol=1e-02):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
