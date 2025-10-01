## [Triton 3.3 Performance Regression on Small Gemms](https://github.com/triton-lang/triton/issues/7096)

Reproducer in [performance-7096/test.py](./performance-7096/test.py) with Triton v3.4.0 [testing.do_bench](https://github.com/triton-lang/triton/blob/v3.4.0/python/triton/testing.py)

```
GPU: NVIDIA GeForce RTX 4090
Triton version: 3.1.0
512x512: 0.0124ms
1024x1024: 0.0210ms
1536x1536: 0.0673ms
2048x2048: 0.1181ms
4096x4096: 0.8580ms
```

```
GPU: NVIDIA GeForce RTX 4090
Triton version: 3.4.0
512x512: 0.0137ms
1024x1024: 0.0225ms
1536x1536: 0.0711ms
2048x2048: 0.1222ms
4096x4096: 0.8852ms
```

Fix use cubin with triton_runner in [fix.py:67](./performance-7096/fix.py#L67)

## [Higher shared_memory usage in Triton 3.3](https://github.com/triton-lang/triton/issues/7268)

Reproducer on NVIDIA GeForce RTX 4090

[high_usage-7268/v3.2.0_cache/_bwd_kernel.json](./high_usage-7268/v3.2.0_cache/_bwd_kernel.json) has `"shared": 98304` and [high_usage-7268/v3.3x.0_cache/_bwd_kernel.json](./high_usage-7268/v3.3.0_cache/_bwd_kernel.json) has `"shared": 114688`

Fix use cubin with triton_runner in [flash_attn_triton_runner.py:152](./high_usage-7268/fix/flash_attn_triton_runner.py#L152)
