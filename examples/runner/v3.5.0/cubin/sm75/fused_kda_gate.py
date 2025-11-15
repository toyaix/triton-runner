import torch
import torch.nn.functional as F
import triton
import triton_runner
import triton.language as tl
from einops import rearrange
import os
import triton.language.extra.libdevice as tldevice

if os.environ.get('FLA_USE_FAST_OPS', '0') == '1':
    exp = tldevice.fast_expf
    exp2 = tldevice.exp2
    log = tldevice.fast_logf
    log2 = tldevice.fast_log2f
else:
    exp = tl.exp
    exp2 = tl.math.exp2
    log = tl.log
    log2 = tl.log2


device = "cuda" if torch.cuda.is_available() else "cpu"

@triton_runner.jit
def kda_gate_fwd_kernel(
    g, A, y,
    g_bias,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    T,
    H,
    D: tl.constexpr,
    BT: tl.constexpr,
    BD: tl.constexpr,
    HAS_BIAS: tl.constexpr,
):
    i_t, i_h = tl.program_id(0), tl.program_id(1)
    n_t = i_t * BT

    b_a = tl.load(A + i_h).to(tl.float32)
    b_a = -tl.exp(b_a)

    stride_row = H * D
    stride_col = 1

    g_ptr = tl.make_block_ptr(
        base=g + i_h * D,
        shape=(T, D),
        strides=(stride_row, stride_col),
        offsets=(n_t, 0),
        block_shape=(BT, BD),
        order=(1, 0),
    )

    y_ptr = tl.make_block_ptr(
        base=y + i_h * D,
        shape=(T, D),
        strides=(stride_row, stride_col),
        offsets=(n_t, 0),
        block_shape=(BT, BD),
        order=(1, 0),
    )

    b_g = tl.load(g_ptr, boundary_check=(0, 1)).to(tl.float32)

    if HAS_BIAS:
        n_d = tl.arange(0, BD)
        bias_mask = n_d < D
        b_bias = tl.load(g_bias + i_h * D + n_d, mask=bias_mask, other=0.0).to(tl.float32)
        b_g = b_g + b_bias[None, :]

    # softplus(x, beta) = (1/beta) * log(1 + exp(beta * x))
    # When beta * x > threshold, use linear approximation x
    # Use threshold to switch to linear when beta*x > threshold
    g_scaled = b_g * beta
    use_linear = g_scaled > threshold
    sp = tl.where(use_linear, b_g, (1.0 / beta) * log(1.0 + tl.exp(g_scaled)))
    b_y = b_a * sp

    tl.store(y_ptr, b_y.to(y.dtype.element_ty), boundary_check=(0, 1))


def kda_gate_fwd(
    g: torch.Tensor,
    A: torch.Tensor,
    head_k_dim: int,
    g_bias: torch.Tensor | None = None,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> torch.Tensor:
    """
    Forward pass for KDA gate:
      input g: [..., H*D]
      param A: [H] or [1, 1, H, 1]
      beta: softplus beta parameter
      threshold: softplus threshold parameter
      return  : [..., H, D]
    """
    orig_shape = g.shape[:-1]

    g = g.view(-1, g.shape[-1])
    T = g.shape[0]
    HD = g.shape[1]
    H = A.numel()
    assert H * head_k_dim == HD

    y = torch.empty_like(g, dtype=torch.float32)

    def grid(meta): return (triton.cdiv(T, meta['BT']), H)

    kda_gate_fwd_kernel[grid](
        g, A, y, g_bias,
        beta, threshold,
        T, H, head_k_dim,
        BD=triton.next_power_of_2(head_k_dim),
        HAS_BIAS=g_bias is not None,
        BT=64,
        num_warps=4,
        num_stages=2,
        cubin_dir=triton_runner.get_file_dir(__file__),
    )

    y = y.view(*orig_shape, H, head_k_dim)
    return y


class KDAGateFunction(torch.autograd.Function):
    """
    Autograd function for KDA gate computation.

    Supports both formats:
    - Standard: [batch_size, seq_len, num_heads * head_k_dim]
    - vLLM: [num_tokens, num_heads * head_k_dim]
    """

    @staticmethod
    def forward(ctx, g: torch.Tensor, A: torch.Tensor, head_k_dim: int,
                g_bias: torch.Tensor | None = None,
                beta: float = 1.0,
                threshold: float = 20.0) -> torch.Tensor:
        ctx.save_for_backward(g, A)
        ctx.g_bias = g_bias
        ctx.head_k_dim = head_k_dim
        ctx.beta = beta
        ctx.threshold = threshold

        return kda_gate_fwd(g, A, head_k_dim, g_bias, beta, threshold)


def fused_kda_gate(g: torch.Tensor, A: torch.Tensor, head_k_dim: int,
                   g_bias: torch.Tensor | None = None,
                   beta: float = 1.0, threshold: float = 20.0) -> torch.Tensor:
    """
    Fused KDA gate computation with autograd support.

    Supports both formats:
    - Standard: [batch_size, seq_len, num_heads * head_k_dim]
    - vLLM: [num_tokens, num_heads * head_k_dim]

    Args:
        g: Input tensor of shape [..., num_heads * head_k_dim]
        A: Parameter tensor of shape [num_heads] or [1, 1, num_heads, 1]
        head_k_dim: Dimension of each head
        beta: softplus beta parameter
        threshold: softplus threshold parameter

    Returns:
        Output tensor of shape [..., num_heads, head_k_dim]
    """
    return KDAGateFunction.apply(g, A, head_k_dim, g_bias, beta, threshold)


def kda_gate_ref(
    g: torch.Tensor,
    A: torch.Tensor,
    head_k_dim: int,
    g_bias: torch.Tensor | None = None,
    b: torch.Tensor | None = None,
    beta=1.0, threshold=20.0,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """
    Torch reference implementation for KDA gate computation.

    Computes: g = -A.exp().unsqueeze(-1) * softplus(rearrange(g, '... (h d) -> ... h d', d=head_k_dim))

    Supports both formats:
    - Standard: [batch_size, seq_len, num_heads * head_k_dim]
    - vLLM: [num_tokens, num_heads * head_k_dim]

    Args:
        g: Input tensor of shape [..., num_heads * head_k_dim]
        A: Parameter tensor of shape [num_heads] or [1, 1, num_heads, 1]
        g_bias : Optional bias tensor added to g before activation, shape [num_heads * head_k_dim]
        b: Optional tensor to compute sigmoid gate, shape [..., num_heads]
        head_k_dim: Dimension of each head

    Returns:
        Output tensor of shape [..., num_heads, head_k_dim]
    """
    # Rearrange g to separate heads: [..., H*D] -> [..., H, D]
    A = A.view(-1)  # Flatten A to [num_heads] to handle any input shape
    if g_bias is not None:
        g = g + g_bias
    g = rearrange(g, '... (h d) -> ... h d', d=head_k_dim)

    # Apply the gate computation: -A.exp().unsqueeze(-1) * softplus(g)
    # A: [H] -> [H, 1] for broadcasting
    A_exp = -A.float().exp().unsqueeze(-1)  # [H, 1]
    g_softplus = F.softplus(g.float(), beta, threshold)      # [..., H, D]

    return A_exp * g_softplus, b.float().sigmoid() if b is not None else None

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

    torch.testing.assert_close(ref[0], tri, rtol=1e-4, atol=1e-4)

    print("✅ Test passed for single configuration!")

    from triton_runner.testing import do_bench
    ms = do_bench(lambda: fused_kda_gate(g.clone(), A.clone(), D, g_bias.clone() if g_bias is not None else None), rep=1000)
    print("ms %.3f"%ms)


if __name__ == "__main__":
    test_kda_gate_single()
