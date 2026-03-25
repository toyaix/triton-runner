import torch
import triton
import triton.language as tl
import triton_runner
import triton_runner.language as dl

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton_runner.jit
def load_store_3d_kernel(
    input_ptr, output_ptr,
    D0, D1, D2,
    BLOCK_D0: tl.constexpr, BLOCK_D1: tl.constexpr, BLOCK_D2: tl.constexpr,
):
    pid0 = tl.program_id(axis=0)
    pid1 = tl.program_id(axis=1)
    pid2 = tl.program_id(axis=2)

    offs_d0 = pid0 * BLOCK_D0 + tl.arange(0, BLOCK_D0)
    offs_d1 = pid1 * BLOCK_D1 + tl.arange(0, BLOCK_D1)
    offs_d2 = pid2 * BLOCK_D2 + tl.arange(0, BLOCK_D2)

    # 3D index: offs_d0[:, None, None] * D1 * D2 + offs_d1[None, :, None] * D2 + offs_d2[None, None, :]
    offs = offs_d0[:, None, None] * (D1 * D2) + offs_d1[None, :, None] * D2 + offs_d2[None, None, :]
    mask = (offs_d0[:, None, None] < D0) & (offs_d1[None, :, None] < D1) & (offs_d2[None, None, :] < D2)

    block = tl.load(input_ptr + offs, mask=mask)

    # ===== DEBUG START =====
    dl.dump(block)
    # ===== DEBUG END =====

    tl.store(output_ptr + offs, block, mask=mask)


def solve(input: torch.Tensor, output: torch.Tensor, D0: int, D1: int, D2: int):
    BLOCK_D0 = 4
    BLOCK_D1 = 8
    BLOCK_D2 = 16
    grid = (triton.cdiv(D0, BLOCK_D0), triton.cdiv(D1, BLOCK_D1), triton.cdiv(D2, BLOCK_D2))

    dump_tensor = torch.empty((BLOCK_D0 * BLOCK_D1, BLOCK_D2), dtype=input.dtype, device=input.device)

    load_store_3d_kernel[grid](
        input, output,
        D0, D1, D2,
        BLOCK_D0=BLOCK_D0, BLOCK_D1=BLOCK_D1, BLOCK_D2=BLOCK_D2,
        dump_tensor=dump_tensor,
    )
    triton_runner.color_print.blue_print(f"debug {dump_tensor}")
    dump_torch = input[:BLOCK_D0, :BLOCK_D1, :BLOCK_D2].reshape(BLOCK_D0 * BLOCK_D1, BLOCK_D2)
    max_diff = torch.max(torch.abs(dump_torch - dump_tensor))
    triton_runner.color_print.yellow_print(f"The maximum difference between torch and dump is {max_diff}")


if __name__ == "__main__":
    D0, D1, D2 = 10, 20, 30
    a = torch.randn((D0, D1, D2), device=DEVICE)
    output = torch.empty_like(a)
    solve(a, output, D0, D1, D2)
    if torch.allclose(output, a):
        print("Triton and Torch match")
    else:
        print("Triton and Torch differ")
