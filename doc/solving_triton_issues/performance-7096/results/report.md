# Cross-Version Performance Report

**Kernel:** `bench_kernel.py` (issue #7096 reproducer)
**GPU:** Tesla T4 (sm75)
**Versions:** 3.1.0, 3.4.0
**Date:** 2026-04-30
**Benchmark:** `triton_runner.testing.do_bench` (warmup=25ms, rep=100ms, return_mode="mean")
**Cubins:** pre-compiled, loaded via `triton_runner.jit` + `cubin_dir`

## Results

| Problem | **3.1.0** | **3.4.0** | Speedup |
|---------|----------|----------|----------|
| 512³ | 0.0359 | 1.4978 | 0.024x |
| 1024³ | 0.1575 | 13.7889 | 0.011x |
| 1536³ | 0.4485 | 49.6177 | 0.009x |
| 2048³ | 0.7452 | 120.6656 | 0.006x |
| 4096³ | 7.1007 | 973.4100 | 0.007x |

| Env | cubin size |
|-----|-----------|
| triton-v3-1-0 | 62K |
| triton-v3-4-0 | 440K |

*Baseline: 3.1.0. Benchmarked with `repro_cubin.py`.*

