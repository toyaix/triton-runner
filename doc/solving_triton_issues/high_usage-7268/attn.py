# Higher shared_memory usage in Triton 3.3
# https://github.com/triton-lang/triton/issues/7268

from flash_attn_triton import flash_attn_func
import torch

# set seed
torch.random.manual_seed(0)
batch_size = 1
nheads = 4
d = 64
seqlen = 16
dtype = torch.bfloat16
q = torch.randn([batch_size, seqlen, nheads, d], dtype=dtype, device="cuda") * 5
k, v = [
    torch.randn([batch_size, seqlen, nheads, d], dtype=dtype, device="cuda") * 3
    for _ in range(2)
]
q.requires_grad_(True)
k.requires_grad_(True)
v.requires_grad_(True)
out = flash_attn_func(q, k, v)
g = torch.randn_like(out)
out.backward(g)
