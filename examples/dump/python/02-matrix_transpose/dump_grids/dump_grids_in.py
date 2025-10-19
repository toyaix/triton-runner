import torch
import triton
import triton.language as tl
import triton_runner
import triton_runner.language as dl

@triton_runner.jit
def matrix_transpose_kernel(input_ptr, output_ptr, rows, cols, BLOCK_SIZE: tl.constexpr):
    row_index = tl.program_id(axis=0)
    col_index = tl.program_id(axis=1)
    offs_row = row_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_col = col_index * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    old_offs = offs_row[:, None] * cols + offs_col[None, :]
    mask = (offs_row[:, None] < rows) & (offs_col[None, :] < cols)
    block = tl.load(input_ptr + old_offs, mask=mask)

    # ===== DEBUG START =====
    dl.dump_grids(block)
    # ===== DEBUG END =====

    transposed_block = tl.trans(block)
    new_block = offs_col[:, None] * rows + offs_row[None, :]
    tl.store(output_ptr + new_block, transposed_block, mask=mask.T)


def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    grid = lambda meta: (triton.cdiv(rows, meta['BLOCK_SIZE']), triton.cdiv(cols, meta['BLOCK_SIZE']))

    BLOCK_SIZE = 64
    block_shape = [BLOCK_SIZE, BLOCK_SIZE]
    pad_n_elements = triton_runner.torch_utils.get_pad_n_elements(input, block_shape)
    dump_tensor = torch.empty(pad_n_elements, dtype=torch.float32, device=input.device)

    matrix_transpose_kernel[grid](
        input, output,
        rows, cols,
        BLOCK_SIZE=BLOCK_SIZE,
        dump_tensor=dump_tensor,
    )
    triton_runner.color_print.blue_print(f"debug {dump_tensor}")
    # is same (triton.cdiv(rows, BLOCK_SIZE), triton.cdiv(cols, BLOCK_SIZE))
    grid_dim = triton_runner.torch_utils.get_grid_dim([rows, cols], block_shape)
    block_reshape = dump_tensor.reshape(*grid_dim, *block_shape)
    block_permute = block_reshape.permute(0, 2, 1, 3)
    reshape_tensor = block_permute.reshape(grid_dim[0] * BLOCK_SIZE, grid_dim[1] * BLOCK_SIZE)
    dump_torch = input
    max_diff = torch.max(torch.abs(dump_torch - reshape_tensor[:rows,:cols]))
    triton_runner.color_print.yellow_print(f"The maximum difference between torch and dump is {max_diff}")


if __name__ == "__main__":
    rows, cols = 1312, 1221
    a = torch.randn((rows, cols), device='cuda')
    torch_output = a.T
    triton_output = torch.empty(torch_output.shape, device='cuda')
    solve(a, triton_output, rows, cols)
    if torch.allclose(triton_output, torch_output):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
