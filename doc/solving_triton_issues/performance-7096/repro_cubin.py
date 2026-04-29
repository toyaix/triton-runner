"""Reproduce performance-7096 benchmark from saved cubins.

Loads pre-compiled cubins via ``triton_runner.jit`` + ``cubin_dir`` and benchmarks
with ``triton_runner.testing.do_bench``.

Usage::

    python repro_cubin.py                          # benchmark both versions
    python repro_cubin.py --version 3.1.0          # benchmark v3.1.0 only
    python repro_cubin.py --version 3.4.0          # benchmark v3.4.0 only
"""

import argparse
import os

import torch
import triton
from triton import language as tl
import triton_runner
from triton_runner.testing import do_bench

HERE = os.path.dirname(os.path.abspath(__file__))
CUBINS_DIR = os.path.join(HERE, "results", "cubins")


@triton_runner.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    pass


def matmul(a, b, cubin_dir):
    """Run matmul using pre-compiled cubin from cubin_dir."""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=torch.float16)
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=64, BLOCK_SIZE_N=128, BLOCK_SIZE_K=64,
        GROUP_SIZE_M=8, num_stages=4, num_warps=4,
        cubin_dir=cubin_dir,
    )
    return c


def main():
    parser = argparse.ArgumentParser(
        description="Reproduce performance-7096 benchmark from saved cubins")
    parser.add_argument("--version", default="",
                        choices=["3.1.0", "3.4.0", ""],
                        help="Triton version to benchmark (default: both)")
    args = parser.parse_args()

    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"Triton host version: {triton.__version__}")
    print()

    versions = [args.version] if args.version else ["3.1.0", "3.4.0"]
    sizes = [512, 1024, 1536, 2048, 4096]

    for ver in versions:
        env_name = f"triton-v{ver.replace('.', '-')}"
        cubin_dir = os.path.join(CUBINS_DIR, env_name)
        cubin_file = os.path.join(cubin_dir, "matmul_kernel.cubin")
        json_file = os.path.join(cubin_dir, "matmul_kernel.json")
        if not (os.path.exists(cubin_file) and os.path.exists(json_file)):
            print(f"[SKIP] {env_name}: cubin not found at {cubin_dir}")
            print(f"       Run: python -m triton_runner.bench.cross_version "
                  f"-k bench_kernel.py --save-cubin")
            continue

        print(f"--- triton-v{ver} (cubin: {os.path.getsize(cubin_file)} bytes) ---")
        for size in sizes:
            a = torch.randn(size, size, device="cuda", dtype=torch.float16)
            b = torch.randn(size, size, device="cuda", dtype=torch.float16)

            # correctness check
            expected = a @ b
            result = matmul(a, b, cubin_dir)
            max_diff = (result - expected).abs().max().item()
            torch.cuda.synchronize()

            # benchmark
            avg = do_bench(lambda: matmul(a, b, cubin_dir))
            print(f"  {size}x{size}: {avg:.4f} ms  (max diff: {max_diff:.4f})")
        print()


if __name__ == "__main__":
    main()
