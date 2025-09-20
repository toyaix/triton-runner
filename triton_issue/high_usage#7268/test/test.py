from flash_attn_triton_test import _flash_attn_backward
import torch
import math
import triton


batch_size, nheads, d, seqlen = 1, 4, 64, 16
torch.random.manual_seed(0)

dtype = torch.bfloat16
q, k, v, o, do = [
    torch.randn([batch_size, seqlen, nheads, d], dtype=dtype, device="cuda")
    for _ in range(5)
]
seqlen_q_rounded = math.ceil(seqlen / 128) * 128
lse = torch.empty((batch_size, nheads, seqlen_q_rounded), device=q.device, dtype=torch.float32)
with torch.inference_mode():
    dq = torch.empty_like(q)
    dk = torch.empty_like(k)
    dv = torch.empty_like(v)
    _flash_attn_backward(
        do,
        q,
        k,
        v,
        o,
        lse,
        dq,
        dk,
        dv
    )
    print(do)
