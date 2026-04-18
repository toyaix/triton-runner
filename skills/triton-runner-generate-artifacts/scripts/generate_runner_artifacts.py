#!/usr/bin/env python3
import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import torch
import triton
import triton.language as tl
import triton_runner
from triton.backends.compiler import GPUTarget
from triton.runtime.driver import driver
from triton.runtime.jit import MockTensor


TEXT_EXTS = {"ttir", "ttgir", "llir", "ptx"}
SUPPORTED_CAPABILITIES = (75, 80, 86, 90, 120)


triton_runner.configure_jit_backend()


def repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def current_triton_version() -> str:
    return triton.__version__


def default_capability() -> int:
    device = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device)
    return major * 10 + minor


def version_root(version: str) -> Path:
    return repo_root() / "examples" / "runner" / f"v{version}"


@contextmanager
def override_target(capability: int):
    original = driver.active.get_current_target
    driver.active.get_current_target = lambda: GPUTarget("cuda", capability, 32)
    try:
        yield
    finally:
        driver.active.get_current_target = original


def to_text(module):
    if isinstance(module, str):
        return module
    if isinstance(module, (bytes, bytearray, memoryview)):
        return bytes(module).decode("utf-8")
    return str(module)


def to_bytes(module):
    if isinstance(module, memoryview):
        return module.tobytes()
    if isinstance(module, bytearray):
        return bytes(module)
    if isinstance(module, bytes):
        return module
    raise TypeError(f"Unsupported binary module type: {type(module)!r}")


def to_metadata_dict(metadata):
    if hasattr(metadata, "_asdict"):
        return metadata._asdict()
    if isinstance(metadata, dict):
        return metadata
    return vars(metadata)


def write_artifact(compiled, out_dir: Path, stem: str, ext: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = compiled.asm[ext]
    path = out_dir / f"{stem}.{ext}"
    if ext in TEXT_EXTS:
        path.write_text(to_text(payload))
    else:
        path.write_bytes(to_bytes(payload))

    if ext in {"llir", "ptx", "cubin"}:
        metadata_path = out_dir / f"{stem}.json"
        metadata_path.write_text(json.dumps(to_metadata_dict(compiled.metadata), default=vars))


def compile_basic(capability: int):
    @triton.jit
    def matmul_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
    ):
        pid_n = tl.program_id(axis=0)
        pid_m = tl.program_id(axis=1)

        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

        a_ptrs = a_ptr + offs_m[:, None] * stride_am
        b_ptrs = b_ptr + offs_n[None, :] * stride_bn

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        for k in range(K):
            a = tl.load(a_ptrs + k * stride_ak)
            b = tl.load(b_ptrs + k * stride_bk)
            accumulator += a * b

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
        tl.store(c_ptrs, accumulator, mask=c_mask)

    M, K, N = 512, 1024, 256
    grid = lambda meta: (triton.cdiv(N, meta["BLOCK_SIZE_N"]), triton.cdiv(M, meta["BLOCK_SIZE_M"]))
    with override_target(capability):
        return matmul_kernel.warmup(
            MockTensor(torch.float32, [M, K]),
            MockTensor(torch.float32, [K, N]),
            MockTensor(torch.float32, [M, N]),
            M,
            N,
            K,
            K,
            1,
            N,
            1,
            N,
            1,
            BLOCK_SIZE_M=16,
            BLOCK_SIZE_N=16,
            grid=grid,
        )


def compile_dot(capability: int):
    @triton.jit
    def matmul_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_an,
        stride_bn,
        stride_bk,
        stride_cm,
        stride_ck,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
    ):
        pid_k = tl.program_id(axis=0)
        pid_m = tl.program_id(axis=1)

        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_k = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)
        offs_n = tl.arange(0, BLOCK_SIZE_N)

        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an
        b_ptrs = b_ptr + offs_n[:, None] * stride_bn + offs_k[None, :] * stride_bk

        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)
        for n in range(0, tl.cdiv(N, BLOCK_SIZE_N)):
            max_idx = N - n * BLOCK_SIZE_N
            a = tl.load(a_ptrs + n * BLOCK_SIZE_N * stride_an, mask=offs_n[None, :] < max_idx, other=0.0)
            b = tl.load(b_ptrs + n * BLOCK_SIZE_N * stride_bn, mask=offs_n[:, None] < max_idx, other=0.0)
            accumulator = tl.dot(a, b, acc=accumulator)

        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_k[None, :] * stride_ck
        c_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        tl.store(c_ptrs, accumulator, mask=c_mask)

    M, N, K = 8192, 6144, 4096
    grid = lambda meta: (triton.cdiv(K, meta["BLOCK_SIZE_K"]), triton.cdiv(M, meta["BLOCK_SIZE_M"]))
    with override_target(capability):
        return matmul_kernel.warmup(
            MockTensor(torch.float16, [M, N]),
            MockTensor(torch.float16, [N, K]),
            MockTensor(torch.float32, [M, K]),
            M,
            N,
            K,
            N,
            1,
            K,
            1,
            K,
            1,
            BLOCK_SIZE_M=128,
            BLOCK_SIZE_K=64,
            BLOCK_SIZE_N=64,
            grid=grid,
        )


def compile_tma(capability: int):
    @triton.jit
    def matmul_kernel_make_tensor_desciptor(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
    ):
        pid_m = tl.program_id(axis=0)
        pid_k = tl.program_id(axis=1)

        a_desc = tl.make_tensor_descriptor(
            a_ptr,
            shape=[M, N],
            strides=[N, 1],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        )
        b_desc = tl.make_tensor_descriptor(
            b_ptr,
            shape=[N, K],
            strides=[K, 1],
            block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
        )
        c_desc = tl.make_tensor_descriptor(
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

        accumulator = accumulator.to(tl.float16)
        c_desc.store([pid_m * BLOCK_SIZE_M, pid_k * BLOCK_SIZE_K], accumulator)

    M, N, K = 1024, 512, 256

    def alloc_fn(size, alignment, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)
    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_SIZE_M"]), triton.cdiv(K, meta["BLOCK_SIZE_K"]))
    with override_target(capability):
        return matmul_kernel_make_tensor_desciptor.warmup(
            MockTensor(torch.float8_e5m2, [M, N]),
            MockTensor(torch.float8_e5m2, [N, K]),
            MockTensor(torch.float16, [M, K]),
            M,
            N,
            K,
            BLOCK_SIZE_M=128,
            BLOCK_SIZE_K=64,
            BLOCK_SIZE_N=64,
            grid=grid,
        )


def generate_sm75(root: Path):
    compiled = compile_basic(75)
    write_artifact(compiled, root / "ttir" / "matmul", "matmul_kernel", "ttir")
    write_artifact(compiled, root / "ttgir" / "sm75", "matmul_kernel", "ttgir")
    write_artifact(compiled, root / "llir" / "sm75", "matmul_kernel", "llir")
    write_artifact(compiled, root / "ptx" / "sm75", "matmul_kernel", "ptx")
    write_artifact(compiled, root / "cubin" / "sm75", "matmul_kernel", "cubin")


def generate_sm80_or_sm86(root: Path, capability: int):
    compiled = compile_dot(capability)
    sm_dir = f"sm{capability}"
    write_artifact(compiled, root / "ttir" / "matmul-with-dot", "matmul_kernel", "ttir")
    write_artifact(compiled, root / "ttgir" / sm_dir, "matmul_kernel", "ttgir")
    write_artifact(compiled, root / "llir" / sm_dir, "matmul_kernel", "llir")
    write_artifact(compiled, root / "ptx" / sm_dir, "matmul_kernel", "ptx")
    write_artifact(compiled, root / "cubin" / sm_dir, "matmul_kernel", "cubin")


def generate_sm90_or_sm120(root: Path, capability: int):
    compiled = compile_tma(capability)
    sm_dir = f"sm{capability}"
    stem = "matmul_kernel_make_tensor_desciptor"
    write_artifact(compiled, root / "ttir" / "matmul-with-tma", stem, "ttir")
    write_artifact(compiled, root / "ttgir" / sm_dir, stem, "ttgir")
    write_artifact(compiled, root / "llir" / sm_dir, stem, "llir")
    write_artifact(compiled, root / "ptx" / sm_dir, stem, "ptx")
    write_artifact(compiled, root / "cubin" / sm_dir, stem, "cubin")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("capabilities", nargs="*", type=int)
    parser.add_argument("--version", default=current_triton_version())
    args = parser.parse_args()

    root = version_root(args.version)
    if not root.exists():
        raise FileNotFoundError(f"Missing version directory: {root}")

    capabilities = args.capabilities or [default_capability()]
    generators = {
        75: lambda: generate_sm75(root),
        80: lambda: generate_sm80_or_sm86(root, 80),
        86: lambda: generate_sm80_or_sm86(root, 86),
        90: lambda: generate_sm90_or_sm120(root, 90),
        120: lambda: generate_sm90_or_sm120(root, 120),
    }

    for capability in capabilities:
        if capability not in SUPPORTED_CAPABILITIES:
            raise ValueError(f"Unsupported capability: {capability}")
        print(f"Generating artifacts for sm{capability} on Triton {triton.__version__}")
        generators[capability]()


if __name__ == "__main__":
    main()
