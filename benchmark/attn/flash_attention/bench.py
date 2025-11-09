from triton_runner.bench.utils import benchmark
import torch
import triton

from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.nn.functional import scaled_dot_product_attention as sdpa


from triton_runner.bench.attn.triton_fused_attention import (
    attention_opt as triton_tutorial_FA2_opt,
)

from typing import Callable
from contextlib import nullcontext

class Operator():

    DEFAULT_METRICS = ["walltime"]

    def get_input_iter(self, BATCH=256, H=128, N_CTX=8, N_CTX_KV=8, D_HEAD=128):
        dtype = torch.bfloat16
        self.device = "cuda"
        q = torch.randn(
            (BATCH, H, N_CTX, D_HEAD),
            dtype=dtype,
            device=self.device,
        )
        k = torch.randn(
            (BATCH, H, N_CTX_KV, D_HEAD),
            dtype=dtype,
            device=self.device,
        )
        v = torch.randn(
            (BATCH, H, N_CTX_KV, D_HEAD),
            dtype=dtype,
            device=self.device,
        )
        self.sm_scale = 1.0 / (D_HEAD**0.5)
        self.causal = False
        self.native_sdpa = True
        self.pt2_sdpa = False
        yield tuple([q, k, v])

    @benchmark("torch_aten")
    def aten(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Callable:
        def _inner():
            seq_len = q.shape[2]
            M = torch.tril(torch.ones((seq_len, seq_len), device=self.device))
            p = torch.matmul(q, k.transpose(2, 3)) * self.sm_scale
            if self.causal:
                p[:, :, M == 0] = float("-inf")
            p = torch.softmax(p.float(), dim=-1).to(q.dtype)
            # p = torch.exp(p)
            ref_out = torch.matmul(p, v)
            return ref_out

        return _inner

    @benchmark("torch_sdpa")
    def sdpa(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Callable:
        def sdpa_flash_attention(q, k, v):
            cxt = (
                nullcontext()
                if self.native_sdpa
                else sdpa_kernel([SDPBackend.FLASH_ATTENTION])
            )
            with cxt:
                sdpa_impl = (
                    torch.compile(
                        sdpa,
                        fullgraph=True,
                        backend="inductor",
                        mode="max-autotune",
                    )
                    if self.pt2_sdpa
                    else sdpa
                )
                return sdpa_impl(
                    q,
                    k,
                    v,
                    is_causal=self.causal,
                    scale=self.sm_scale,
                )

        return lambda: sdpa_flash_attention(
            q,
            k,
            v,
        )

    @benchmark("triton_tutorial_flash_v2")
    def triton_tutorial_flash_v2(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Callable:
        # includes base (default scheduling) + opt (optimized loop scheduling based on heuristics)
        return lambda: triton_tutorial_FA2_opt(
            q, k, v, self.causal, self.sm_scale, "base_opt"
        )

    @benchmark("triton_tutorial_flash_v2_tma")
    def triton_tutorial_flash_v2_tma(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Callable:
        # autotune TMA/CompPipe
        return lambda: triton_tutorial_FA2_opt(
            q, k, v, self.causal, self.sm_scale, "tma"
        )

    @benchmark("triton_tutorial_flash_v2_ws")
    def triton_tutorial_flash_v2_ws(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Callable:
        # autotune WarpSpec/CompPipe
        return lambda: triton_tutorial_FA2_opt(
            q, k, v, self.causal, self.sm_scale, "ws"
        )

    @benchmark("triton_tutorial_flash_v2_tma_ws")
    def triton_tutorial_flash_v2_tma_ws(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Callable:
        # autotune TMA/WarpSpec/CompPipe
        return lambda: triton_tutorial_FA2_opt(
            q, k, v, self.causal, self.sm_scale, "tma_ws"
        )

    @benchmark("triton_tutorial_flash_v2_tma_ws_persistent")
    def triton_tutorial_flash_v2_tma_ws_persistent(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> Callable:
        # autotune TMA/WarpSpec/CompPipe/Persistent
        return lambda: triton_tutorial_FA2_opt(
            q, k, v, self.causal, self.sm_scale, "tma_ws_persistent"
        )


def has_warp_spec():
    import triton.language as tl

    return hasattr(tl, "async_task")


def has_assume():
    import triton.language as tl

    return hasattr(tl, "assume")


def has_new_tma():
    import triton
    import triton.language as tl

    return hasattr(triton, "set_allocator") and hasattr(tl, "make_tensor_descriptor")

def cuda_capability_geq(major, minor=0):
    """
    Determines whether we have compute capability >= (major, minor) and
    returns this as a constexpr boolean. This can be used for guarding
    inline asm implementations that require a certain compute capability.
    """
    return torch.cuda.get_device_capability() >= (major, minor)

def cuda_capability_eq(major, minor=0):
    """
    Determines whether we have compute capability >= (major, minor) and
    returns this as a constexpr boolean. This can be used for guarding
    inline asm implementations that require a certain compute capability.
    """
    return torch.cuda.get_device_capability() == (major, minor)

if __name__ == "__main__":
    op = Operator()
    op.aten()
    op.sdpa()
    if has_assume():
        op.triton_tutorial_flash_v2()
    if has_new_tma() and cuda_capability_geq(9):
        op.triton_tutorial_flash_v2_tma()
    if has_warp_spec() and cuda_capability_geq(9) and not cuda_capability_eq(12):
        op.triton_tutorial_flash_v2_ws()
    if has_warp_spec() and has_new_tma() and cuda_capability_geq(9) and not cuda_capability_eq(12):
        op.triton_tutorial_flash_v2_tma_ws()
        op.triton_tutorial_flash_v2_tma_ws_persistent()
