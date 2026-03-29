import argparse
import dataclasses
import os
import statistics
import time
from collections.abc import Callable

os.environ.setdefault("TRITON_TVM_FFI", "1")

import torch
import triton
import triton.language as tl


_ORIGINAL_TRITON_JIT = triton.jit
FP8_DTYPE = getattr(torch, "float8_e5m2", None)
SCALAR_BLOCKS = (16, 16)
DOT_BLOCKS = (128, 64, 64)
TMA_BLOCKS = (128, 64, 64)


@dataclasses.dataclass(frozen=True)
class ArchConfig:
    variant: str
    input_dtype: torch.dtype
    output_dtype: torch.dtype
    atol: float


@dataclasses.dataclass(frozen=True)
class LaunchPlan:
    grid: tuple[int, int]
    compiled_bound_args: tuple[object, ...]
    direct_bound_args: tuple[object, ...]
    plain_launch: Callable[[], None]
    runner_warmup: Callable[[], object]


@_ORIGINAL_TRITON_JIT
def plain_scalar_matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
                               stride_cn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    pid_n = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am
    b_ptrs = b_ptr + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(K):
        acc += tl.load(a_ptrs + k * stride_ak) * tl.load(b_ptrs + k * stride_bk)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


@_ORIGINAL_TRITON_JIT
def plain_dot_matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_an, stride_bn, stride_bk, stride_cm,
                            stride_ck, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                            BLOCK_SIZE_N: tl.constexpr):
    pid_k = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an
    b_ptrs = b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    for n in range(tl.cdiv(N, BLOCK_SIZE_N)):
        remaining = N - n * BLOCK_SIZE_N
        a = tl.load(a_ptrs + n * BLOCK_SIZE_N * stride_an, mask=offs_n[None, :] < remaining, other=0.0)
        b = tl.load(b_ptrs + n * BLOCK_SIZE_N * stride_bn, mask=offs_n[:, None] < remaining, other=0.0)
        acc = tl.dot(a, b, acc=acc)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_k[None, :] * stride_ck
    mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    tl.store(c_ptrs, acc, mask=mask)


@_ORIGINAL_TRITON_JIT
def plain_tma_matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                            BLOCK_SIZE_K: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[N, K], strides=[K, 1], block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K])
    c_desc = tl.make_tensor_descriptor(c_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K])
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    for n in range(tl.cdiv(N, BLOCK_SIZE_N)):
        a = a_desc.load([pid_m * BLOCK_SIZE_M, n * BLOCK_SIZE_N])
        b = b_desc.load([n * BLOCK_SIZE_N, pid_k * BLOCK_SIZE_K])
        acc = tl.dot(a, b, acc=acc)

    c_desc.store([pid_m * BLOCK_SIZE_M, pid_k * BLOCK_SIZE_K], acc.to(tl.float16))


import triton_runner

triton_runner.configure_jit_backend()


@triton.jit
def runner_scalar_matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_ak, stride_bk, stride_bn, stride_cm,
                                stride_cn, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr):
    pid_n = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am
    b_ptrs = b_ptr + offs_n[None, :] * stride_bn
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    for k in range(K):
        acc += tl.load(a_ptrs + k * stride_ak) * tl.load(b_ptrs + k * stride_bk)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc, mask=mask)


@triton.jit
def runner_dot_matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, stride_am, stride_an, stride_bn, stride_bk, stride_cm,
                             stride_ck, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
                             BLOCK_SIZE_N: tl.constexpr):
    pid_k = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)

    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
    offs_n = tl.arange(0, BLOCK_SIZE_N)
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an
    b_ptrs = b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    for n in range(tl.cdiv(N, BLOCK_SIZE_N)):
        remaining = N - n * BLOCK_SIZE_N
        a = tl.load(a_ptrs + n * BLOCK_SIZE_N * stride_an, mask=offs_n[None, :] < remaining, other=0.0)
        b = tl.load(b_ptrs + n * BLOCK_SIZE_N * stride_bn, mask=offs_n[:, None] < remaining, other=0.0)
        acc = tl.dot(a, b, acc=acc)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_k[None, :] * stride_ck
    mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
    tl.store(c_ptrs, acc, mask=mask)


@triton.jit
def runner_tma_matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
                             BLOCK_SIZE_K: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    a_desc = tl.make_tensor_descriptor(a_ptr, shape=[M, N], strides=[N, 1], block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N])
    b_desc = tl.make_tensor_descriptor(b_ptr, shape=[N, K], strides=[K, 1], block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K])
    c_desc = tl.make_tensor_descriptor(c_ptr, shape=[M, K], strides=[K, 1], block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K])
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    for n in range(tl.cdiv(N, BLOCK_SIZE_N)):
        a = a_desc.load([pid_m * BLOCK_SIZE_M, n * BLOCK_SIZE_N])
        b = b_desc.load([n * BLOCK_SIZE_N, pid_k * BLOCK_SIZE_K])
        acc = tl.dot(a, b, acc=acc)

    c_desc.store([pid_m * BLOCK_SIZE_M, pid_k * BLOCK_SIZE_K], acc.to(tl.float16))


def _arch_configs() -> dict[int, ArchConfig]:
    configs = {
        75: ArchConfig("scalar", torch.float32, torch.float32, 1e-2),
        80: ArchConfig("dot", torch.float16, torch.float32, 0.125),
        86: ArchConfig("dot", torch.float16, torch.float32, 0.125),
    }
    if FP8_DTYPE is not None:
        configs[90] = ArchConfig("tma", FP8_DTYPE, torch.float16, 0.125)
        configs[120] = ArchConfig("tma", FP8_DTYPE, torch.float16, 0.125)
    return configs


def _resolve_arch_config() -> tuple[int, ArchConfig]:
    props = torch.cuda.get_device_properties(torch.device("cuda"))
    sm = props.major * 10 + props.minor
    config = _arch_configs().get(sm)
    if config is None:
        supported = ", ".join(f"sm{arch}" for arch in sorted(_arch_configs()))
        raise RuntimeError(f"No python repro config for sm{sm}. Supported architectures: {supported}")
    return sm, config


def _enable_tma_allocator() -> None:
    def alloc_fn(size, alignment, stream):
        del alignment, stream
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)


def _bench_host_us(fn: Callable[[], None], *, warmup: int, iters: int, repeats: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1e6 / iters)
    return statistics.median(samples)


def _reference(a: torch.Tensor, b: torch.Tensor, config: ArchConfig) -> torch.Tensor:
    if config.variant == "tma":
        return torch.matmul(a.to(torch.float16), b.to(torch.float16))
    return torch.matmul(a, b)


def _validate_outputs(ref: torch.Tensor, atol: float, *outputs: torch.Tensor) -> None:
    ref = ref.float()
    for name, out in zip(("Native Triton", "TVM-Triton", "Direct TVM-FFI"), outputs, strict=True):
        if not torch.allclose(out.float(), ref, atol=atol, rtol=0):
            raise AssertionError(f"{name} output mismatch.")


def _make_direct_launch(compiled_kernel, grid: tuple[int, int], direct_bound_args: tuple[object, ...]) -> Callable[[], None]:
    launcher = compiled_kernel._get_launcher()
    direct_args = tuple(arg for entry, arg in zip(launcher._signature, direct_bound_args, strict=True) if not entry.is_constexpr)
    grid_x, grid_y = grid

    def launch() -> None:
        launcher._tvm_func(launcher._registry_handle, grid_x, grid_y, 1, *direct_args)

    return launch


def _make_compiled_launch(compiled_kernel, grid: tuple[int, int], compiled_bound_args: tuple[object, ...]) -> Callable[[], None]:
    launcher = compiled_kernel[(grid[0], grid[1], 1)]

    def launch() -> None:
        launcher(*compiled_bound_args)

    return launch


def _make_scalar_plan(a: torch.Tensor, b: torch.Tensor, c_plain: torch.Tensor, c_runner: torch.Tensor,
                      c_direct: torch.Tensor, m: int, n: int, k: int) -> LaunchPlan:
    block_m, block_n = SCALAR_BLOCKS
    grid = (triton.cdiv(n, block_n), triton.cdiv(m, block_m))
    call_args_plain = (
        a, b, c_plain, m, n, k,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c_plain.stride(0), c_plain.stride(1),
    )
    call_args_runner = (
        a, b, c_runner, m, n, k,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c_runner.stride(0), c_runner.stride(1),
    )
    direct_bound_args = (
        a, b, c_direct, m, n, k,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c_direct.stride(0), c_direct.stride(1),
        block_m, block_n,
    )
    compiled_bound_args = (
        a, b, c_runner, m, n, k,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c_runner.stride(0), c_runner.stride(1),
        block_m, block_n,
    )
    meta = {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n}

    def plain_launch() -> None:
        plain_scalar_matmul_kernel[grid](*call_args_plain, **meta)

    def runner_warmup():
        return runner_scalar_matmul_kernel.warmup(*call_args_runner, grid=grid, **meta)

    return LaunchPlan(grid, compiled_bound_args, direct_bound_args, plain_launch, runner_warmup)


def _make_dot_plan(a: torch.Tensor, b: torch.Tensor, c_plain: torch.Tensor, c_runner: torch.Tensor,
                   c_direct: torch.Tensor, m: int, n: int, k: int) -> LaunchPlan:
    block_m, block_k, block_n = DOT_BLOCKS
    grid = (triton.cdiv(n, block_k), triton.cdiv(m, block_m))
    call_args_plain = (
        a, b, c_plain, m, k, n,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c_plain.stride(0), c_plain.stride(1),
    )
    call_args_runner = (
        a, b, c_runner, m, k, n,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c_runner.stride(0), c_runner.stride(1),
    )
    direct_bound_args = (
        a, b, c_direct, m, k, n,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c_direct.stride(0), c_direct.stride(1),
        block_m, block_k, block_n,
    )
    compiled_bound_args = (
        a, b, c_runner, m, k, n,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c_runner.stride(0), c_runner.stride(1),
        block_m, block_k, block_n,
    )
    meta = {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_K": block_k, "BLOCK_SIZE_N": block_n}

    def plain_launch() -> None:
        plain_dot_matmul_kernel[grid](*call_args_plain, **meta)

    def runner_warmup():
        return runner_dot_matmul_kernel.warmup(*call_args_runner, grid=grid, **meta)

    return LaunchPlan(grid, compiled_bound_args, direct_bound_args, plain_launch, runner_warmup)


def _make_tma_plan(a: torch.Tensor, b: torch.Tensor, c_plain: torch.Tensor, c_runner: torch.Tensor,
                   c_direct: torch.Tensor, m: int, n: int, k: int) -> LaunchPlan:
    block_m, block_n, block_k = TMA_BLOCKS
    grid = (triton.cdiv(m, block_m), triton.cdiv(n, block_k))
    call_args_plain = (a, b, c_plain, m, k, n)
    call_args_runner = (a, b, c_runner, m, k, n)
    direct_bound_args = (a, b, c_direct, m, k, n, block_m, block_n, block_k)
    compiled_bound_args = (a, b, c_runner, m, k, n, block_m, block_n, block_k)
    meta = {"BLOCK_SIZE_M": block_m, "BLOCK_SIZE_N": block_n, "BLOCK_SIZE_K": block_k}

    def plain_launch() -> None:
        plain_tma_matmul_kernel[grid](*call_args_plain, **meta)

    def runner_warmup():
        return runner_tma_matmul_kernel.warmup(*call_args_runner, grid=grid, **meta)

    return LaunchPlan(grid, compiled_bound_args, direct_bound_args, plain_launch, runner_warmup)


def _make_launch_plan(config: ArchConfig, a: torch.Tensor, b: torch.Tensor, c_plain: torch.Tensor,
                      c_runner: torch.Tensor, c_direct: torch.Tensor, m: int, n: int, k: int) -> LaunchPlan:
    if config.variant == "scalar":
        return _make_scalar_plan(a, b, c_plain, c_runner, c_direct, m, n, k)
    if config.variant == "dot":
        return _make_dot_plan(a, b, c_plain, c_runner, c_direct, m, n, k)
    return _make_tma_plan(a, b, c_plain, c_runner, c_direct, m, n, k)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce host launch latency for plain Triton vs TVM-Triton vs direct launch on Python kernels."
    )
    parser.add_argument("--m", type=int, default=1)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--k", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2000)
    parser.add_argument("--iters", type=int, default=20000)
    parser.add_argument("--repeats", type=int, default=7)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to run this benchmark.")

    device = torch.device("cuda")
    props = torch.cuda.get_device_properties(device)
    sm, config = _resolve_arch_config()
    if config.variant == "tma":
        _enable_tma_allocator()

    a = torch.randn((args.m, args.k), device=device, dtype=torch.float32).to(config.input_dtype)
    b = torch.randn((args.k, args.n), device=device, dtype=torch.float32).to(config.input_dtype)
    c_plain = torch.empty((args.m, args.n), device=device, dtype=config.output_dtype)
    c_runner = torch.empty_like(c_plain)
    c_direct = torch.empty_like(c_plain)

    plan = _make_launch_plan(config, a, b, c_plain, c_runner, c_direct, args.m, args.n, args.k)
    compiled_runner = plan.runner_warmup()
    if hasattr(compiled_runner, "result"):
        compiled_runner = compiled_runner.result()
    compiled_launch = _make_compiled_launch(compiled_runner, plan.grid, plan.compiled_bound_args)
    direct_launch = _make_direct_launch(compiled_runner, plan.grid, plan.direct_bound_args)

    plan.plain_launch()
    compiled_launch()
    direct_launch()
    torch.cuda.synchronize()
    _validate_outputs(_reference(a, b, config), config.atol, c_plain, c_runner, c_direct)

    triton_us = _bench_host_us(plan.plain_launch, warmup=args.warmup, iters=args.iters, repeats=args.repeats)
    tvm_triton_us = _bench_host_us(compiled_launch, warmup=args.warmup, iters=args.iters, repeats=args.repeats)
    direct_us = _bench_host_us(direct_launch, warmup=args.warmup, iters=args.iters, repeats=args.repeats)

    grid_x, grid_y = plan.grid
    print(f"device: {props.name} (cc {props.major}.{props.minor})")
    print(f"kernel: sm{sm} python matmul (no cubin_dir)")
    print(f"problem: A=({args.m}, {args.k}) B=({args.k}, {args.n}) C=({args.m}, {args.n}) grid=({grid_x}, {grid_y}, 1)")
    print(f"measure: host launch latency, median over {args.repeats} repeats")
    print(f"Triton: {triton_us:.3f} us")
    print(f"TVM-Triton (CompiledTVMFFIKernel.__getitem__/run): {tvm_triton_us:.3f} us")
    print(f"direct launch: {direct_us:.3f} us")
    print(f"TVM-Triton - direct launch: {tvm_triton_us - direct_us:.3f} us")
    print(f"Triton - direct launch: {triton_us - direct_us:.3f} us")


if __name__ == "__main__":
    main()
