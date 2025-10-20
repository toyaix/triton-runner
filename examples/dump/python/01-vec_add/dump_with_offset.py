import triton
import triton.language as tl
import torch
import triton_runner
import triton_runner.language as dl

if triton.__version__ in ["3.2.0", "3.1.0", "3.0.0"]:
    DEVICE = torch.cuda.current_device()
else:
    DEVICE = triton.runtime.driver.active.get_active_torch_device()

@triton_runner.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    # There are multiple 'programs' processing different data. We identify which program
    # we are here:
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    # This program will process inputs that are offset from the initial data.
    # For instance, if you had a vector of length 256 and block_size of 64, the programs
    # would each access the elements [0:64, 64:128, 128:192, 192:256].
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses.
    mask = offsets < n_elements
    # Load x and y from DRAM, masking out any extra elements in case the input is not a
    # multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)

    # ===== DEBUG START =====
    # dl.dump(y, 0, (1, 0, 0))
    dl.dump(y, BLOCK_SIZE, 1)
    # ===== DEBUG END =====

    output = x + y
    # Write x + y back to DRAM.
    tl.store(output_ptr + offsets, output, mask=mask)


# %%
# Let's also declare a helper function to (1) allocate the `z` tensor
# and (2) enqueue the above kernel with appropriate grid/block sizes:


def add(x: torch.Tensor, y: torch.Tensor):
    # We need to preallocate the output.
    output = torch.empty_like(x)

    n_elements = output.numel()
    # The SPMD launch grid denotes the number of kernel instances that run in parallel.
    # It is analogous to CUDA launch grids. It can be either Tuple[int], or Callable(metaparameters) -> Tuple[int].
    # In this case, we use a 1D grid where the size is the number of blocks:
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    BLOCK_SIZE = 1024
    dump_tensor = torch.empty_like(output)

    # NOTE:
    #  - Each torch.tensor object is implicitly converted into a pointer to its first element.
    #  - `triton.jit`'ed functions can be indexed with a launch grid to obtain a callable GPU kernel.
    #  - Don't forget to pass meta-parameters as keywords arguments.
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=BLOCK_SIZE,
                     dump_tensor=dump_tensor,
    )
    dump_torch = y
    grid_0 = 1
    block_start = BLOCK_SIZE * grid_0
    dump_offset = BLOCK_SIZE
    triton_runner.color_print.blue_print(f"debug {dump_tensor[dump_offset: dump_offset + BLOCK_SIZE]}")
    max_diff = torch.max(torch.abs(dump_torch[block_start: block_start + BLOCK_SIZE] - dump_tensor[dump_offset: dump_offset + BLOCK_SIZE]))
    triton_runner.color_print.yellow_print(f"The maximum difference between torch and dump is {max_diff}")
    # We return a handle to z but, since `torch.cuda.synchronize()` hasn't been called, the kernel is still
    # running asynchronously at this point.
    return output


# %%
# We can now use the above function to compute the element-wise sum of two `torch.tensor` objects and test its correctness:

if __name__ == "__main__":
    size = 98432
    x = torch.rand(size, device=DEVICE)
    y = torch.rand(size, device=DEVICE)
    torch_output = x + y
    triton_output = add(x, y)
    if torch.allclose(triton_output, torch_output):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")
