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
    transposed_block = tl.trans(block)

    # ===== DEBUG START =====
    dl.dump_grids(transposed_block)
    # ===== DEBUG END =====

    new_block = offs_col[:, None] * rows + offs_row[None, :]
    tl.store(output_ptr + new_block, transposed_block, mask=mask.T)


def solve(input: torch.Tensor, output: torch.Tensor, rows: int, cols: int):
    grid = lambda meta: (triton.cdiv(rows, meta['BLOCK_SIZE']), triton.cdiv(cols, meta['BLOCK_SIZE']))

    BLOCK_SIZE = 64
    import math
    new_shape = tuple(triton.cdiv(dim, block) * block for dim, block in zip(input.shape, [BLOCK_SIZE, BLOCK_SIZE]))
    dump_tensor = torch.empty(math.prod(new_shape), dtype=torch.float32, device=input.device)

    matrix_transpose_kernel[grid](
        input, output,
        rows, cols,
        BLOCK_SIZE=BLOCK_SIZE,
        dump_tensor=dump_tensor,
    )
    triton_runner.color_print.blue_print(f"debug {dump_tensor}")
    grid_dim = tuple(triton.cdiv(dim, block) for dim, block in zip(input.shape, [BLOCK_SIZE, BLOCK_SIZE]))
    block_reshape = dump_tensor.reshape(*grid_dim, BLOCK_SIZE, BLOCK_SIZE)
    block_permute = block_reshape.permute(1, 2, 0, 3)
    reshape_tensor = block_permute.reshape(grid_dim[1] * BLOCK_SIZE, grid_dim[0] * BLOCK_SIZE)
    dump_torch = a.T
    max_diff = torch.max(torch.abs(dump_torch - reshape_tensor[:cols,:rows]))
    triton_runner.color_print.yellow_print(f"The maximum difference between torch and dump is {max_diff}")


if __name__ == "__main__":
    rows, cols = 2036, 765
    a = torch.randn((rows, cols), device='cuda')
    torch_output = a.T
    triton_output = torch.empty(torch_output.shape, device='cuda')
    solve(a, triton_output, rows, cols)
    if torch.allclose(triton_output, torch_output):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
