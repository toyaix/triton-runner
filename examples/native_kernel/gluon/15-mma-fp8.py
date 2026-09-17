"""
FP8 Matmul on sm120 (MMAv2)
===========================

sm120 (consumer Blackwell, e.g. the RTX 5090) tensor cores are fed by the same
Ampere-style ``mma.sync`` (MMAv2) instructions as sm80/sm89. There is no Hopper
WGMMA and no datacenter-Blackwell tcgen05/TMEM, which is why tutorials
05/06/10/11 of the upstream series cannot run on sm120. For fp8, sm120
provides:

- the plain ``mma.sync`` fp8 shapes (e4m3/e5m2) ``m16n8k32``, and
- the scaled ``mma.sync`` variant for microscaling (MX) fp8 reached through
  ``tl.dot_scaled``, which Triton emits *only* for capability 120 (on sm100 it
  uses tcgen05 instead).

This tutorial demonstrates:

1. A plain fp8 (e4m3/e5m2) blocked matmul, in Gluon, using TMA loads and the
   ``MMAv2`` wrapper from ``07-persistence.py``. The wrapper is dtype-generic:
   for fp8 the dot operands use kWidth=4 and the compiler emits
   ``mma.sync.m16n8k32...e4m3`` / ``...e5m2``.
2. An MXFP8 (fp8 + ue8m0 scales) matmul through ``tl.dot_scaled``, which
   lowers to ``mma.sync...kind::mxf8f6f4.block_scale...``. The ue8m0 scales
   are loaded as raw uint8 tiles.

   Note that part 2 uses a plain ``@triton.jit`` kernel: in Triton 3.8.0 the
   Gluon compilation pipeline does not run ``accelerate-matmul``, so a
   ``dot_scaled`` in a Gluon kernel keeps its blocked encoding all the way to
   the LLVM conversion, which cannot lower it. Only the regular Triton
   frontend reaches the scaled ``mma.sync`` path on sm120.

The fp4 (e2m1) sibling of part 2 is covered in ``16-mma-fp4.py``, which reuses
the ``matmul_mx_kernel`` defined below.
"""

import pytest
import torch
import triton
import importlib
import triton.language as tl
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.hopper import tma, mbarrier, fence_async_shared

# Re-use the MMAv2 (mma.sync) wrapper from the persistence tutorial.
t7 = importlib.import_module("07-persistence")


def get_capability():
    return torch.cuda.get_device_capability()


def is_sm89_or_newer():
    # fp8 mma.sync requires sm89 or newer.
    return torch.cuda.is_available() and get_capability() >= (8, 9)


def is_sm100_or_newer():
    # dot_scaled lowers to tcgen05 on sm100 and to scaled mma.sync on sm120.
    return torch.cuda.is_available() and get_capability()[0] >= 10


if __name__ == "__main__" and not is_sm89_or_newer():
    raise RuntimeError("This tutorial requires sm89 or newer NVIDIA GPU")

# %%
# Part 1: Plain fp8 matmul with TMA and the MMAv2 wrapper
# -------------------------------------------------------
#
# Structurally this is the single-buffered implicit-GEMM loop from
# ``13-conv-im2col.py``: TMA both operand tiles into shared memory through one
# mbarrier, issue the MMA, and store the accumulator through a TMA epilogue.
# The only fp8-specific parts are the operand dtypes and the tensor
# descriptors: NVMMASharedLayout requires the innermost block dimension to
# cover at least one swizzle unit, which for 8-bit elements and the default
# 128-byte swizzle means BLOCK_K >= 128.


@gluon.jit
def matmul_fp8_kernel(a_desc, b_desc, c_desc, MMAImpl: gl.constexpr, num_warps: gl.constexpr):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype
    K = a_desc.shape[1]

    pid_m = gl.program_id(0)
    pid_n = gl.program_id(1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    a_smem = gl.allocate_shared_memory(dtype, a_desc.block_type.shape, a_desc.layout)
    b_smem = gl.allocate_shared_memory(dtype, b_desc.block_type.shape, b_desc.layout)
    tma_bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
    mbarrier.init(tma_bar, count=1)

    # The wrapper computes kWidth = 32 // bitwidth = 4 for fp8 operands, so the
    # shared-memory tiles load straight into m16n8k32 dot operands.
    mma = MMAImpl.initialize(dtype, BLOCK_M, BLOCK_N, num_warps)
    phase = 0
    for k in range(0, K, BLOCK_K):
        mbarrier.expect(tma_bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes)
        tma.async_load(a_desc, [off_m, k], tma_bar, a_smem)
        tma.async_load(b_desc, [k, off_n], tma_bar, b_smem)
        mbarrier.wait(tma_bar, phase)
        mma = mma.wait_num_outstanding(0)
        mma = mma.issue_async_mma(a_smem, b_smem)
        phase ^= 1
    mbarrier.invalidate(tma_bar)

    mma = mma.wait_num_outstanding(0)
    c_smem = gl.allocate_shared_memory(c_desc.dtype, c_desc.block_type.shape, c_desc.layout)
    c, mma = mma.take_result()
    c_smem.store(c.to(c_desc.dtype))
    fence_async_shared()
    tma.async_store(c_desc, [off_m, off_n], c_smem)
    tma.store_wait(pendings=0)


# torch fp8 dtype name -> gluon dtype name
_TORCH_TO_GLUON_FP8 = {
    "float8_e4m3fn": "float8e4nv",
    "float8_e5m2": "float8e5",
}


def matmul_fp8(A, B, C, BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, num_warps=4):
    assert A.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    M, K = A.shape
    Kb, N = B.shape
    assert K == Kb

    gl_dtype = getattr(gl, _TORCH_TO_GLUON_FP8[str(A.dtype).split('.')[1]])
    a_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl_dtype)
    b_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], gl_dtype)
    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float16)
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K], a_layout)
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_K, BLOCK_N], b_layout)
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], c_layout)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_fp8_kernel[grid](a_desc, b_desc, c_desc, t7.MMAv2, num_warps=num_warps)


@pytest.mark.parametrize("dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
@pytest.mark.parametrize("M, N, K", [(128, 128, 512), (256, 128, 256)])
@pytest.mark.skipif(not is_sm89_or_newer(), reason="Requires fp8 tensor cores (sm89+)")
def test_matmul_fp8(dtype, M, N, K):
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda").to(dtype)
    B = torch.randn(K, N, device="cuda").to(dtype)
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)
    matmul_fp8(A, B, C)
    ref = A.float() @ B.float()
    # fp8 mantissa is 3 bits (e4m3) / 2 bits (e5m2); the absolute error of the
    # K-long dot product grows with sqrt(K).
    tol = 0.35 * (K**0.5)
    torch.testing.assert_close(C.float(), ref, rtol=0.1, atol=tol)


# %%
# Part 2: MXFP8 matmul via tl.dot_scaled (plain Triton)
# ------------------------------------------------------
#
# Microscaling formats multiply 32-element groups of the operands by a
# power-of-two ue8m0 scale byte. In Triton, ``tl.dot_scaled`` takes the operand
# tiles plus the scale tiles as raw uint8. The operands below are loaded
# through regular global loads and the sm120 lowering turns the operation into
# the block-scaled ``mma.sync`` directly.
#
# A is [M, K] (K-major), B is stored [N, K] but addressed as [K, N], and the
# scale tiles are [M, K // 32] and [N, K // 32]. Note that rhs_scale is NOT
# transposed. The kernel is format-generic so that ``16-mma-fp4.py`` can reuse
# it with FORMAT="e2m1", where the operands become uint8 with two fp4 values
# packed along K and the instruction becomes m16n8k64.


@triton.jit
def matmul_mx_kernel(a_ptr, b_ptr, a_scale_ptr, b_scale_ptr, c_ptr,  #
                     K,  #
                     FORMAT: tl.constexpr,  #
                     BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    PACK: tl.constexpr = 2 if FORMAT == "e2m1" else 1
    GROUP_SIZE: tl.constexpr = 32  # ue8m0 scale group

    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None]
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]
    offs_bn_rows = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))[:, None]
    offs_ak = tl.arange(0, BLOCK_K // PACK)[None, :]
    offs_bk = tl.arange(0, BLOCK_K // PACK)[:, None]
    offs_sk = tl.arange(0, BLOCK_K // GROUP_SIZE)[None, :]

    # A and B are contiguous with K // PACK as the inner dimension.
    a_ptrs = a_ptr + offs_am * (K // PACK) + offs_ak
    b_ptrs = b_ptr + offs_bk + offs_bn * (K // PACK)
    a_scale_ptrs = a_scale_ptr + offs_am * (K // GROUP_SIZE) + offs_sk
    b_scale_ptrs = b_scale_ptr + offs_bn_rows * (K // GROUP_SIZE) + offs_sk

    offs_cm = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M))[:, None]
    offs_cn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N))[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        a_scale = tl.load(a_scale_ptrs)
        b_scale = tl.load(b_scale_ptrs)
        acc = tl.dot_scaled(a, a_scale, FORMAT, b, b_scale, FORMAT, acc)
        a_ptrs += BLOCK_K // PACK
        b_ptrs += BLOCK_K // PACK
        a_scale_ptrs += BLOCK_K // GROUP_SIZE
        b_scale_ptrs += BLOCK_K // GROUP_SIZE

    tl.store(c_ptr + offs_cm * BLOCK_N + offs_cn, acc.to(tl.float16))


def matmul_mx(a, a_scale, b, b_scale, fmt, BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, num_warps=4):
    M, K_packed = a.shape
    N = b.shape[0]
    K = K_packed * (2 if fmt == "e2m1" else 1)
    c = torch.empty(M, N, device=a.device, dtype=torch.float16)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_mx_kernel[grid](a, b, a_scale, b_scale, c, K, fmt, BLOCK_M, BLOCK_N, BLOCK_K, num_warps=num_warps)
    return c


# %%
# Testing helpers: dequantize the MX operands on the host. A ue8m0 byte is
# 2 ** (x - 127).


def dequant_e8m0(scale):
    return torch.exp2(scale.to(torch.float32) - 127.0)


def make_mxfp8_ref(a, a_scale, b, b_scale):
    a_f = a.to(torch.float32) * dequant_e8m0(a_scale).repeat_interleave(32, dim=1)
    b_f = b.to(torch.float32) * dequant_e8m0(b_scale).repeat_interleave(32, dim=1)
    return a_f @ b_f.T


@pytest.mark.parametrize("fmt", ["e4m3", "e5m2"])
@pytest.mark.parametrize("M, N, K", [(128, 128, 512), (256, 128, 256)])
@pytest.mark.skipif(not is_sm100_or_newer(), reason="Requires dot_scaled tensor cores (sm100+)")
def test_matmul_mxfp8(fmt, M, N, K):
    torch.manual_seed(0)
    torch_dtype = {"e4m3": torch.float8_e4m3fn, "e5m2": torch.float8_e5m2}[fmt]
    # Keep the values small so the scaled products stay well inside range.
    a = (torch.randn(M, K, device="cuda") * 0.5).to(torch_dtype)
    b = (torch.randn(N, K, device="cuda") * 0.5).to(torch_dtype)
    a_scale = torch.full((M, K // 32), 127, dtype=torch.uint8, device="cuda")
    b_scale = torch.full((N, K // 32), 127, dtype=torch.uint8, device="cuda")
    c = matmul_mx(a, a_scale, b, b_scale, fmt)
    ref = make_mxfp8_ref(a, a_scale, b, b_scale)
    torch.testing.assert_close(c.float(), ref, rtol=0.1, atol=0.35 * (K**0.5))


# %%
# A quick benchmark comparing the fp8 paths against fp16 on one tile shape.

if __name__ == "__main__":
    M = N = K = 4096
    BLOCK = 128

    A16 = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B16 = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C16 = torch.empty(M, N, device="cuda", dtype=torch.float16)
    if is_sm89_or_newer():
        A8 = A16.to(torch.float8_e4m3fn)
        B8 = B16.to(torch.float8_e4m3fn)
        C8 = torch.empty(M, N, device="cuda", dtype=torch.float16)
        a_scale = torch.full((M, K // 32), 127, dtype=torch.uint8, device="cuda")
        b_scale = torch.full((N, K // 32), 127, dtype=torch.uint8, device="cuda")

        print("Benchmarking fp8 matmul")
        print("=======================")
        ms = triton.testing.do_bench(lambda: t7.matmul_pipelined(A16, B16, C16, BLOCK, BLOCK, 64, 2, 4))
        print(f"matmul_pipelined (fp16): {t7.get_flops(ms, M, N, K):8.2f} tflops/s")
        ms = triton.testing.do_bench(lambda: matmul_fp8(A8, B8, C8, BLOCK, BLOCK, 128, 4))
        print(f"matmul_fp8 (e4m3):       {t7.get_flops(ms, M, N, K):8.2f} tflops/s")
        if is_sm100_or_newer():
            ms = triton.testing.do_bench(lambda: matmul_mx(A8, a_scale, B8, b_scale, "e4m3", BLOCK, BLOCK, 128))
            print(f"matmul_mx (mxfp8):       {t7.get_flops(ms, M, N, K):8.2f} tflops/s")
