import os

import torch
import torch.nn.functional as F

from fla.ops.kda.gate import kda_gate_ref
from gate import fused_kda_gate
from fla.utils import assert_close, device


device = "cuda" if torch.cuda.is_available() else "cpu"

def test_kda_gate_single():
    """Run single configuration of kda gate test (B=1, T=2, H=2, D=12, use_bias=False)"""
    B, T, H, D, use_bias = (1, 2, 2, 12, False)

    print(f"Running test for B={B}, T={T}, H={H}, D={D}, use_bias={use_bias}")

    # torch.manual_seed(42)

    g = torch.randn(B, T, H * D, dtype=torch.float32)
    g = g * 30
    A = torch.log(torch.randn(1, 1, H, 1, dtype=torch.float32).uniform_(1, 16))
    g_bias = torch.randn(H * D, dtype=torch.float32) if use_bias else None

    g, A = map(lambda x: x.to(device).requires_grad_(True), (g, A))
    if g_bias is not None:
        g_bias = g_bias.to(device).requires_grad_(True)

    ref = kda_gate_ref(g.clone(), A.clone(), D, g_bias.clone() if g_bias is not None else None)
    tri = fused_kda_gate(g.clone(), A.clone(), D, g_bias.clone() if g_bias is not None else None)

    assert_close('o', ref, tri, 1e-4)

    print("✅ Test passed for single configuration!")


if __name__ == "__main__":
    test_kda_gate_single()
