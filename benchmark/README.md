# benchmark command

```shell
python benchmark/launch_latency/bench.py
python benchmark/launch_latency/repro_cubin_matmul.py
python benchmark/launch_latency/repro_python_matmul.py

python benchmark/matmul/mma/bench.py
```

`launch_latency/bench.py` 只在 Triton `v3.3.0+` 上运行；更低版本会直接跳过。

`repro_cubin_matmul.py` 走 `cubin_dir` 路径。

`repro_python_matmul.py` 走正常的 Python/JIT 路径，不传 `cubin_dir`。
