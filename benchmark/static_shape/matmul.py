from triton_runner.bench.matmul.kernels import matmul_kernel, runner_matmul_kernel
from triton_runner.bench.utils import benchmark
import torch
import triton


class Operator:
    DEFAULT_METRICS = ["walltime"]

    def get_input_iter(self, M=8192, N=4096, K=4096):
        a = torch.randn((M, N), device="cuda", dtype=torch.float32)
        b = torch.randn((N, K), device="cuda", dtype=torch.float32)
        c = torch.randn((M, K), device="cuda", dtype=torch.float32)
        stride_am, stride_an = N, 1
        stride_bn, stride_bk = K, 1
        stride_cm, stride_ck = K, 1
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K = 128, 64, 64
        yield tuple([
            a, b, c, M, N, K, stride_am, stride_an, stride_bn, stride_bk, stride_cm, stride_ck, BLOCK_SIZE_M,
            BLOCK_SIZE_N, BLOCK_SIZE_K
        ])

    @benchmark("triton")
    def matmul_triton_kernel(self, *args):
        M, K = args[3], args[5]
        BLOCK_SIZE_M, BLOCK_SIZE_K = args[-3], args[-1]
        grid = (triton.cdiv(K, BLOCK_SIZE_K), triton.cdiv(M, BLOCK_SIZE_M))
        return lambda: matmul_kernel[grid](*args)

    @benchmark("triton_runner")
    def matmul_triton_runner_kernel(self, *args):
        M, K = args[3], args[5]
        BLOCK_SIZE_M, BLOCK_SIZE_K = args[-3], args[-1]
        grid = (triton.cdiv(K, BLOCK_SIZE_K), triton.cdiv(M, BLOCK_SIZE_M))
        return lambda: runner_matmul_kernel[grid](*args)

    @benchmark("matmul_triton_runner_compiled")
    def matmul_triton_runner_compiled_kernel(self, *args):
        M, K = args[3], args[5]
        BLOCK_SIZE_M, BLOCK_SIZE_K = args[-3], args[-1]
        grid = (triton.cdiv(K, BLOCK_SIZE_K), triton.cdiv(M, BLOCK_SIZE_M))
        return lambda: runner_matmul_kernel[grid](*args).run()


def check_triton():
    op = Operator()
    args = list(op.get_input_iter(9434, 2422, 4233))[0]
    torch_output = torch.matmul(args[0], args[1])
    op.matmul_triton_kernel(args, enable_benchmark=False)()
    triton_output = args[2]
    if torch.allclose(triton_output, torch_output, atol=1e-1, rtol=1e-2):
        print("✅ Triton and Torch match")
    else:
        print("❌ Triton and Torch differ")


def check_triton_runner():
    op = Operator()
    args = list(op.get_input_iter(9434, 2422, 4233))[0]
    torch_output = torch.matmul(args[0], args[1])
    op.matmul_triton_runner_kernel(args, enable_benchmark=False)()
    triton_output = args[2]
    if torch.allclose(triton_output, torch_output, atol=1e-1, rtol=1e-2):
        print("✅ Triton runner and Torch match")
    else:
        print("❌ Triton runner and Torch differ")


def triton_runner_compiled():
    op = Operator()
    args = list(op.get_input_iter(9434, 2422, 4233))[0]
    torch_output = torch.matmul(args[0], args[1])
    op.matmul_triton_runner_compiled_kernel(args, enable_benchmark=False)()
    triton_output = args[2]
    if torch.allclose(triton_output, torch_output, atol=1e-1, rtol=1e-2):
        print("✅ Triton runner compiled and Torch match")
    else:
        print("❌ Triton runner compiled and Torch differ")


if __name__ == "__main__":
    check_triton()
    check_triton_runner()
    triton_runner_compiled()
    op = Operator()
    op.matmul_triton_kernel()
    op.matmul_triton_runner_kernel()
    op.matmul_triton_runner_compiled_kernel()
