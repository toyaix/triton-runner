"""
FP4 (MXFP4) Matmul on sm120
===========================

This tutorial is the fp4 sibling of part 2 of ``15-mma-fp8.py`` and reuses its
format-generic ``matmul_mx_kernel``. On sm120, ``tl.dot_scaled`` with e2m1
operands lowers to the block-scaled fp4 instruction

    mma.sync.aligned.m16n8k64.row.col.kind::mxf4nvf4.block_scale.scale_vec::2X.f32.e2m1.e2m1.f32.ue8m0

i.e. an ``m16n8k64`` tensor-core op (double the K of the fp8 ``m16n8k32``
variant) with one ue8m0 exponent byte scaling each 32-element group. Triton
emits this path only for capability 120; on sm100 the same ``tl.dot_scaled``
uses tcgen05 instead.

sm120 restrictions enforced by the compiler:

- fp4 x fp4 only; mixed fp8 x fp4 is not supported.
- fp4 operands must be packed into uint8 along K (``lhs_k_pack=rhs_k_pack=
  True``, the defaults) with the first element in the low nibble.
- both ue8m0 scale tensors must be present (loaded as raw uint8).

The e2m1 format has a 1-bit mantissa: the magnitudes are exactly
{0, 0.5, 1, 1.5, 2, 3, 4, 6}, so host-side quantization and dequantization are
simple LUT operations.
"""

import pytest
import torch
import triton
import importlib
from triton.experimental.gluon import language as gl

# Re-use the format-generic dot_scaled kernel and helpers from the fp8 tutorial.
t15 = importlib.import_module("15-mma-fp8")
matmul_mx_kernel = t15.matmul_mx_kernel
dequant_e8m0 = t15.dequant_e8m0


def is_sm100_or_newer():
    # dot_scaled lowers to tcgen05 on sm100 and to scaled mma.sync on sm120.
    return torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 10


if __name__ == "__main__" and not is_sm100_or_newer():
    raise RuntimeError("This tutorial requires a Blackwell (sm100+) NVIDIA GPU")

# %%
# Host-side e2m1 helpers. ``_E2M1_LUT`` maps a magnitude index (the low 3 bits
# of a nibble) to its value; bit 3 of the nibble is the sign. Quantization
# rounds to the nearest grid magnitude via the midpoint thresholds.

_E2M1_LUT = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
_E2M1_MIDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0])


def quantize_e2m1(x):
    """Pack a float tensor [M, K] into e2m1 nibbles, uint8 [M, K // 2]."""
    mids = _E2M1_MIDS.to(x.device)
    mag = x.abs().clamp(max=6.0)
    idx = torch.searchsorted(mids, mag.contiguous()).to(torch.uint8)
    nib = idx | ((x < 0).to(torch.uint8) << 3)
    lo, hi = nib[..., 0::2], nib[..., 1::2]
    return (lo | (hi << 4)).contiguous()


def dequant_e2m1(packed):
    """Unpack uint8 [M, K // 2] into float [M, K]."""
    lut = _E2M1_LUT.to(packed.device)
    lo = lut[(packed & 0x7).long()] * (1.0 - 2.0 * ((packed & 0x8) >> 3).float())
    hi = lut[((packed >> 4) & 0x7).long()] * (1.0 - 2.0 * ((packed & 0x80) >> 7).float())
    return torch.stack((lo, hi), dim=-1).reshape(packed.shape[0], -1)


def make_mxfp4_ref(a, a_scale, b, b_scale):
    a_f = dequant_e2m1(a) * dequant_e8m0(a_scale).repeat_interleave(32, dim=1)
    b_f = dequant_e2m1(b) * dequant_e8m0(b_scale).repeat_interleave(32, dim=1)
    return a_f @ b_f.T


# %%
# The launcher just wraps ``t15.matmul_mx`` with FORMAT="e2m1"; A and B are
# uint8 [M, K // 2] and the scale tiles are [M, K // 32] / [N, K // 32].


def matmul_mxfp4(a, a_scale, b, b_scale, BLOCK_M=128, BLOCK_N=128, BLOCK_K=128, num_warps=4):
    return t15.matmul_mx(a, a_scale, b, b_scale, "e2m1", BLOCK_M, BLOCK_N, BLOCK_K, num_warps)


# %%
# Because e2m1 products are exact in fp32, the only differences against the
# host reference come from accumulation order and the fp16 store.

@pytest.mark.parametrize("M, N, K", [(128, 128, 512), (256, 128, 256)])
@pytest.mark.skipif(not is_sm100_or_newer(), reason="Requires dot_scaled tensor cores (sm100+)")
def test_matmul_mxfp4_unit_scale(M, N, K):
    torch.manual_seed(0)
    a = quantize_e2m1(torch.randn(M, K, device="cuda"))
    b = quantize_e2m1(torch.randn(N, K, device="cuda"))
    a_scale = torch.full((M, K // 32), 127, dtype=torch.uint8, device="cuda")
    b_scale = torch.full((N, K // 32), 127, dtype=torch.uint8, device="cuda")
    c = matmul_mxfp4(a, a_scale, b, b_scale)
    ref = make_mxfp4_ref(a, a_scale, b, b_scale)
    torch.testing.assert_close(c.float(), ref, rtol=1e-2, atol=1.0)


@pytest.mark.parametrize("M, N, K", [(128, 128, 512)])
@pytest.mark.skipif(not is_sm100_or_newer(), reason="Requires dot_scaled tensor cores (sm100+)")
def test_matmul_mxfp4_random_scales(M, N, K):
    torch.manual_seed(0)
    a = quantize_e2m1(torch.randn(M, K, device="cuda"))
    b = quantize_e2m1(torch.randn(N, K, device="cuda"))
    # ue8m0 exponents in [124, 130] (2^-3 .. 2^3) keep the products in fp16 range.
    a_scale = torch.randint(124, 131, (M, K // 32), dtype=torch.uint8, device="cuda")
    b_scale = torch.randint(124, 131, (N, K // 32), dtype=torch.uint8, device="cuda")
    c = matmul_mxfp4(a, a_scale, b, b_scale)
    ref = make_mxfp4_ref(a, a_scale, b, b_scale)
    torch.testing.assert_close(c.float(), ref, rtol=1e-2, atol=8.0)


# %%
# A quick benchmark against the fp16 and fp8 paths from the earlier tutorials.

if __name__ == "__main__":
    t7 = importlib.import_module("07-persistence")
    M = N = K = 4096
    BLOCK = 128

    A16 = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B16 = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C16 = torch.empty(M, N, device="cuda", dtype=torch.float16)

    a4 = quantize_e2m1(A16)
    b4 = quantize_e2m1(B16)
    a_scale = torch.full((M, K // 32), 127, dtype=torch.uint8, device="cuda")
    b_scale = torch.full((N, K // 32), 127, dtype=torch.uint8, device="cuda")

    print("Benchmarking fp4 matmul")
    print("=======================")
    ms = triton.testing.do_bench(lambda: t7.matmul_pipelined(A16, B16, C16, BLOCK, BLOCK, 64, 2, 4))
    print(f"matmul_pipelined (fp16): {t7.get_flops(ms, M, N, K):8.2f} tflops/s")
    if t15.is_sm89_or_newer():
        A8 = A16.to(torch.float8_e4m3fn)
        B8 = B16.to(torch.float8_e4m3fn)
        C8 = torch.empty(M, N, device="cuda", dtype=torch.float16)
        ms = triton.testing.do_bench(lambda: t15.matmul_fp8(A8, B8, C8, BLOCK, BLOCK, 128, 4))
        print(f"matmul_fp8 (e4m3):       {t7.get_flops(ms, M, N, K):8.2f} tflops/s")
    ms = triton.testing.do_bench(lambda: matmul_mxfp4(a4, a_scale, b4, b_scale, BLOCK, BLOCK, 128))
    print(f"matmul_mxfp4 (e2m1):     {t7.get_flops(ms, M, N, K):8.2f} tflops/s")
