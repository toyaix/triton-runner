import torch
import torch.nn.functional as F
import triton
import triton_runner
import triton.language as tl
from pathlib import Path

from fla.ops.utils.op import log
from fla.utils import autotune_cache_kwargs, input_guard, is_amd

device = torch.cuda.current_device()
capability = torch.cuda.get_device_capability(device)
capability = capability[0] * 10 + capability[1]

BT_LIST_AUTOTUNE = [32, 64, 128]
NUM_WARPS_AUTOTUNE = [2, 4, 8, 16] if is_amd else [4, 8, 16, 32]

cache_dir = Path(triton_runner.get_file_dir(__file__)).parent / f"kda_gate_fwd_kernel_cache_sm{capability}"

@triton_runner.autotune(
    configs=[
        triton.Config({'autotune_cubin_dir': str(p)}) for p in cache_dir.iterdir() if p.is_dir()
    ],
    key=['H', 'D'],
)
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

    @input_guard
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
