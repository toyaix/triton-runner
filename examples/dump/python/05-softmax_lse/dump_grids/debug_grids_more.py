# softmax use log_sum_exp
import torch
import triton
import triton.language as tl
import triton_runner
import triton_runner.language as dl


@triton_runner.jit
def softmax_kernel(
    input_ptr, output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr
):
    input_ptr = input_ptr.to(tl.pointer_type(tl.float32))
    output_ptr = output_ptr.to(tl.pointer_type(tl.float32))
    max_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32) - float("inf")
    log_acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)

    for off in range(0, N, BLOCK_SIZE):
        cols = off + tl.arange(0, BLOCK_SIZE)
        a = tl.load(input_ptr + cols, mask=cols < N, other=-float("inf"))
        block_max = tl.max(a, axis=0)
        max_acc_new = tl.where(max_acc > block_max, max_acc, block_max)

        raw_exp =  tl.math.exp(a - max_acc_new)

        log_acc_new = tl.math.exp(max_acc - max_acc_new) * log_acc + tl.sum(raw_exp, axis=-1)

        log_acc = log_acc_new
        max_acc = max_acc_new

    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < N
    x = tl.load(input_ptr + offset, mask=mask)
    o = tl.math.exp(x - max_acc) / log_acc

    # ===== DEBUG START =====
    dl.dump_grids(max_acc)
    grid = tl.num_programs(axis=0)
    out_padding_shape = BLOCK_SIZE * grid
    dl.dump_grids(log_acc, out_padding_shape)
    # ===== DEBUG END =====

    tl.store(output_ptr + offset, o, mask=mask)

def torch_softmax(input):
    max = input.max()
    sum = (input - max).exp().sum()
    normalize_by_sum = ((input - max).exp()) / sum
    return normalize_by_sum

def solve(input: torch.Tensor, output: torch.Tensor, N: int):
    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE']), )

    BLOCK_SIZE = 32768
    pad_n_elements = triton_runner.torch_utils.get_pad_n_elements(output, [BLOCK_SIZE])
    dump_tensor = torch.empty(2 * pad_n_elements, dtype=torch.float32, device=input.device)

    softmax_kernel[grid](
        input, output, N,
        BLOCK_SIZE=BLOCK_SIZE,
        dump_tensor=dump_tensor,
    )
    triton_runner.color_print.blue_print(f"debug {dump_tensor}")
    max_acc = dump_tensor[0:N]
    log_acc = dump_tensor[-pad_n_elements: -pad_n_elements+N]
    dump_torch = torch_softmax(input)
    max_diff = torch.max(torch.abs(dump_torch - (input - max_acc)/log_acc))
    triton_runner.color_print.yellow_print(f"The maximum difference between torch and dump is {max_diff}")


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
