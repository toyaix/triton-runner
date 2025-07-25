### sm90 (H100, H200, H20, etc.)
```bash
python examples/python_runner/matmul.py

python examples/ttir_runner/matmul.py

python examples/v3.4.0/ttgir_runner/sm90/matmul-with-tma-v4.py

python examples/v3.4.0/llir_runner/sm90/matmul-with-tma-v4.py

# There appears to be a bug in ptxas for this case. The ptxas bundled with NVCC 12.8.93 is not compatible with .version 8.4. Triton 3.3.x is designed to work with NVCC 12.4.99, which is the correct version for targeting .version 8.4.
python examples/v3.4.0/ptx_runner/sm90/matmul-with-tma-v4.py

python examples/cubin_runner/sm90/matmul-with-tma-v3.py
```

### sm80 (A100, A30)
```bash
python examples/python_runner/matmul.py

python examples/ttir_runner/matmul.py

python examples/ttgir_runner/sm80/matmul-with-dot-v2.py

python examples/llir_runner/sm80/matmul-with-dot-v2.py

python examples/ptx_runner/sm80/matmul-with-dot-v2.py

python examples/cubin_runner/sm80/matmul-with-dot-v2.py
```

### sm120 (RTX PRO 6000, RTX 5090, etc.)
```bash
python examples/python_runner/matmul.py

python examples/ttir_runner/matmul.py

python examples/ttgir_runner/sm120/matmul-with-tma-v3.py

python examples/llir_runner/sm120/matmul-with-tma-v3.py

python examples/ptx_runner/sm120/matmul-with-tma-v3.py

python examples/cubin_runner/sm120/matmul-with-tma-v3.py
```

### sm86 (A10, RTX 3090, etc.)
```bash
python examples/python_runner/matmul.py

python examples/ttir_runner/matmul.py

python examples/ttgir_runner/sm86/matmul-with-dot-v2.py

python examples/llir_runner/sm86/matmul-with-dot-v2.py

python examples/ptx_runner/sm86/matmul-with-dot-v2.py

python examples/cubin_runner/sm86/matmul-with-dot-v2.py
```

### sm75 (T4, RTX 2080, etc.)
```bash
python examples/python_runner/matmul.py

python examples/ttir_runner/matmul.py

python examples/ttgir_runner/sm75/matmul.py

python examples/llir_runner/sm75/matmul.py

python examples/ptx_runner/sm75/matmul.py

python examples/cubin_runner/sm75/matmul.py
```
