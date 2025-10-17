import torch
import triton
import triton.language as tl
import triton_runner


@triton_runner.jit
def softmax_kernel(
    input_ptr, output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr
):
    input_ptr = input_ptr.to(tl.pointer_type(tl.float32))
    output_ptr = output_ptr.to(tl.pointer_type(tl.float32))
    _max = tl.zeros([BLOCK_SIZE], dtype=tl.float32) - float("inf")
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(input_ptr + cols, mask=cols < N, other=-float("inf"))
        _max = tl.maximum(a, _max)
    max = tl.max(_max, axis=0)
    _sum = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(input_ptr + cols, mask=cols < N, other=-float("inf"))
        _sum += tl.exp(a - max)
    sum = tl.sum(_sum, axis=0)
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    x = tl.load(input_ptr + offset, mask=mask)
    exp_shifted = tl.exp(x - max)
    normalize_by_sum =  exp_shifted / sum
    tl.store(output_ptr + offset, normalize_by_sum, mask=mask)


def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), )

    BLOCK_SIZE = 32768
    dump_tensor = torch.empty((BLOCK_SIZE), dtype=torch.float32, device=input.device)
    # dump_value can be "%16"(exp_shifted)
    dump_value = "%16"

    softmax_kernel[grid](
        input, output, N,
        BLOCK_SIZE=BLOCK_SIZE,
        ttgir_dir=triton_runner.get_file_dir(__file__),
        dump_tensor=dump_tensor,
        dump_value=dump_value,
    )
    triton_runner.color_print.blue_print(f"debug {dump_tensor}")
    debug_torch = (input - input.max()).exp()
    max_diff = torch.max(torch.abs(debug_torch[:BLOCK_SIZE] - dump_tensor))
    triton_runner.color_print.yellow_print(f"The maximum difference between torch and debug is {max_diff}")


if __name__ == "__main__":
    N = 100000
    input = torch.randn((N), device='cuda')
    torch_output = torch.softmax(input, 0)
    triton_output = torch.empty(torch_output.shape, device='cuda')
    solve(input, triton_output, N)
    if torch.allclose(triton_output, torch_output):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
