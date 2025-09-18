### sm90 (H100, H200, H20, etc.)
```shell
python examples/python_runner/matmul-with-tma-v4.py

python examples/ttir_runner/matmul-with-tma/matmul-with-tma-v4.py

python examples/v3.1.0/ttgir_runner/sm90/matmul-with-dot-v2.py

python examples/v3.1.0/llir_runner/sm90/matmul-with-dot-v2.py

python examples/v3.1.0/ptx_runner/sm90/matmul-with-dot-v2.py

python examples/v3.1.0/cubin_runner/sm90/matmul-with-dot-v2.py
```

### sm80 (A100, A30)
```shell
python examples/python_runner/matmul-with-dot-v2.py

python examples/ttir_runner/matmul-with-dot/matmul-with-dot-v2.py

python examples/v3.1.0/ttgir_runner/sm80/matmul-with-dot-v2.py

python examples/v3.1.0/llir_runner/sm80/matmul-with-dot-v2.py

python examples/ptx_runner/sm80/matmul-with-dot-v2.py

python examples/cubin_runner/sm80/matmul-with-dot-v2.py
```

### sm120 (RTX PRO 6000, RTX 5090, etc.)

**not supported**

### sm86 (A10, RTX 3090, etc.)
```shell
python examples/python_runner/matmul-with-dot-v2.py

python examples/ttir_runner/matmul-with-dot/matmul-with-dot-v2.py

python examples/v3.1.0/ttgir_runner/sm86/matmul-with-dot-v2.py

python examples/v3.1.0/llir_runner/sm86/matmul-with-dot-v2.py

python examples/ptx_runner/sm86/matmul-with-dot-v2.py

python examples/cubin_runner/sm86/matmul-with-dot-v2.py
```

### sm75 (T4, RTX 2080, etc.)

```shell
python examples/python_runner/matmul.py

python examples/ttir_runner/matmul/matmul.py

python examples/v3.1.0/ttgir_runner/sm75/matmul.py

python examples/v3.1.0/llir_runner/sm75/matmul.py

python examples/v3.2.0/ptx_runner/sm75/matmul.py

python examples/cubin_runner/sm75/matmul.py
```