import torch
import triton
import triton.language as tl
import triton_runner

@triton_runner.jit
def matrix_transpose_kernel(input_ptr, output_ptr, rows, cols, BLOCK_SIZE: tl.constexpr):
    row_index = tl.program_id(axis=0)
    col_index = tl.program_id(axis=1)
    offs_row = row_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_col = col_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    old_offs = offs_row[:, None] * cols + offs_col[None, :]
    mask = (offs_row[:, None] < rows) & (offs_col[None, :] < cols)
    block = tl.load(input_ptr + old_offs, mask=mask)
    transposed_block = tl.trans(block)
    new_block = offs_col[:, None] * rows + offs_row[None, :]
    tl.store(output_ptr + new_block, transposed_block, mask=mask.T)


def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    grid = lambda meta: (triton.cdiv(rows, meta['BLOCK_SIZE']), triton.cdiv(cols, meta['BLOCK_SIZE']))

    BLOCK_SIZE = 64
    dump_tensor = torch.empty((BLOCK_SIZE, BLOCK_SIZE), dtype=input.dtype, device=input.device)
    # dump_value can be "%9"(input)
    dump_value = "%25"

    matrix_transpose_kernel[grid](
        input, output,
        rows, cols,
        BLOCK_SIZE=BLOCK_SIZE,
        ttir_dir=triton_runner.get_file_dir(__file__),
        dump_tensor=dump_tensor,
        dump_value=dump_value,
    )
    triton_runner.color_print.blue_print(f"debug {dump_tensor}")
    debug_torch = input
    max_diff = torch.max(torch.abs(debug_torch[:BLOCK_SIZE, :BLOCK_SIZE] - dump_tensor))
    triton_runner.color_print.yellow_print(f"The maximum difference between torch and debug is {max_diff}")

if __name__ == "__main__":
    rows, cols = 104, 78
    a = torch.randn((rows, cols), device='cuda')
    torch_output = a.T
    triton_output = torch.empty(torch_output.shape, device='cuda')
    solve(a, triton_output, rows, cols)
    if torch.allclose(triton_output, torch_output):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
