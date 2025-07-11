# The use of PyTorch in Triton programs is not allowed for the purposes of fair benchmarking.
import triton
import triton.language as tl
import torch

DEVICE = triton.runtime.driver.active.get_active_torch_device()


@triton.jit
def matmul_kernel_make_tensor_desciptor(a_ptr, b_ptr, c_ptr,  #
                                        M, N, K,  #
                                        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                                        BLOCK_SIZE_K: tl.constexpr,  #
                                        ):
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    a_desc = tl._experimental_make_tensor_descriptor(
        a_ptr,
        shape=[M, N],
        strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )
    b_desc = tl._experimental_make_tensor_descriptor(
        b_ptr,
        shape=[N, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )
    c_desc = tl._experimental_make_tensor_descriptor(
        c_ptr,
        shape=[M, K],
        strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    for n in range(tl.cdiv(N, BLOCK_SIZE_N)):
        a = a_desc.load([pid_m * BLOCK_SIZE_M, n * BLOCK_SIZE_N])
        b = b_desc.load([n * BLOCK_SIZE_N, pid_k * BLOCK_SIZE_K])
        accumulator = tl.dot(a, b, acc=accumulator)

    accumulator = accumulator.to(a_desc.dtype)
    c_desc.store([pid_m * BLOCK_SIZE_M, pid_k * BLOCK_SIZE_K], accumulator)


# a_ptr, b_ptr, c_ptr are raw device pointers
def matmul(a, b):
    M, N = a.shape
    N, K = b.shape

    # Leading dimensions must be multiples of 16-byte strides
    # if M % 4 == 0 and N % 4 == 0 and K % 4 == 0:

    # Allocates output.
    c = torch.empty((M, K), device=a.device, dtype=torch.float16)

    # TMA descriptors require a global memory allocation
    def alloc_fn(size, alignment, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)

    from triton_ml_runner.cubin_utils import get_cufunction, cubin_launch
    kernel_name = "matmul_kernel_make_tensor_desciptor"
    import os
    current_dir = os.path.dirname(os.path.abspath(__file__))
    metadata_path = os.path.join(current_dir, f"{kernel_name}.json")
    cubin_path = os.path.join(current_dir, f"{kernel_name}.cubin")
    function = get_cufunction(metadata_path, cubin_path, f"{kernel_name}")

    bound_args = (a, b, c, M, N, K, a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1), 128,
                  64, 64)
    signature_str = "* * * i32 i32 i32 i32 constexpr i32 constexpr i32 constexpr constexpr constexpr constexpr"
    grid = (
        triton.cdiv(M, 128),
        triton.cdiv(K, 64),
    )
    cubin_launch(function, signature_str, bound_args, grid)

    # grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(K, META['BLOCK_SIZE_K']), )
    # matmul_kernel_make_tensor_desciptor[grid](
    #     a, b, c,
    #     M, N, K,
    #     BLOCK_SIZE_M=128,
    #     BLOCK_SIZE_K=64,
    #     BLOCK_SIZE_N=64,
    # )

    return c


M, N, K = 1024, 512, 256

device = torch.cuda.current_device()

a = torch.randn(M, N, device=DEVICE, dtype=torch.float16)
b = torch.randn(N, K, device=DEVICE, dtype=torch.float16)
torch_output = torch.matmul(a, b)
triton_output = matmul(a, b)

if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")
