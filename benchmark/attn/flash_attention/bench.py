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

    def get_input_iter(self):
        dtype = torch.bfloat16
        self.device = "cuda"
        BATCH, H, N_CTX, N_CTX_KV, D_HEAD = 64, 8192, 8, 8, 64
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

    @benchmark("triton_tutorial")
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


if __name__ == "__main__":
    op = Operator()
    op.aten()
    op.sdpa()
    op.triton_tutorial_flash_v2()
