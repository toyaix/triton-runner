## [Triton 3.3 Performance Regression on Small Gemms](https://github.com/triton-lang/triton/issues/7096)

Reproducer

```
GPU: NVIDIA GeForce RTX 4090
Triton version: 3.1.0
512x512: 0.0364ms
1024x1024: 0.0303ms
1536x1536: 0.0662ms
2048x2048: 0.1166ms
4096x4096: 0.8202ms
```

```
GPU: NVIDIA GeForce RTX 4090
Triton version: 3.4.0
512x512: 0.0448ms
1024x1024: 0.0422ms
1536x1536: 0.0712ms
2048x2048: 0.1133ms
4096x4096: 0.8767ms
```

## [Higher shared_memory usage in Triton 3.3](https://github.com/triton-lang/triton/issues/7268)

Reproducer on NVIDIA GeForce RTX 4090

[triton_issue/high_usage#7268/_bwd_kernel_v3.2.0.json](triton_issue/high_usage#7268/_bwd_kernel_v3.2.0.json) has `"shared": 98304` and [triton_issue/high_usage#7268/_bwd_kernel_v3.3.0.json](triton_issue/high_usage#7268/_bwd_kernel_v3.3.0.json) has `"shared": 114688`
