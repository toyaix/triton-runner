### sm90 (H100, H200, H20, etc.)
```bash
python examples/python_runner/matmul.py

python examples/ttir_runner/matmul.py

python examples/v3.2.0/ttgir_runner/sm90/matmul-with-tma-v3.py

python examples/v3.2.0/llir_runner/sm90/matmul-with-tma-v3.py

python examples/v3.2.0/ptx_runner/sm90/matmul-with-tma-v3.py

python examples/cubin_runner/sm90/matmul-with-tma-v3.py
```

### sm80 (A100, A30)
```bash
python examples/python_runner/matmul.py

python examples/v3.2.0/ttir_runner/matmul.py

python examples/v3.2.0/ttgir_runner/sm80/matmul-with-dot-v2.py

python examples/v3.2.0/llir_runner/sm80/matmul-with-dot-v2.py

python examples/ptx_runner/sm80/matmul-with-dot-v2.py

python examples/cubin_runner/sm80/matmul-with-dot-v2.py
```

### sm86 (A10, RTX 3090, etc.)
```bash
python examples/python_runner/matmul.py

python examples/ttir_runner/matmul.py

python examples/v3.2.0/ttgir_runner/sm86/matmul-with-dot-v2.py

python examples/v3.2.0/llir_runner/sm86/matmul-with-dot-v2.py

python examples/v3.2.0/ptx_runner/sm86/matmul-with-dot-v2.py

python examples/cubin_runner/sm86/matmul-with-dot-v2.py
```

### sm75 (T4, RTX 2080, etc.)
```bash
python examples/python_runner/matmul.py

python examples/ttir_runner/matmul.py

python examples/v3.2.0/ttgir_runner/sm75/matmul.py

python examples/v3.2.0/llir_runner/sm75/matmul.py

python examples/v3.2.0/ptx_runner/sm75/matmul.py

python examples/cubin_runner/sm75/matmul.py
```