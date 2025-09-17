
### sm86 (A10, RTX 3090, etc.)
```shell
python examples/python_runner/matmul-with-dot-v2.py

python examples/ttir_runner/matmul-with-dot/matmul-with-dot-v2.py

python examples/v3.1.0/ttgir_runner/sm86/matmul-with-dot-v2.py

python examples/v3.1.0/llir_runner/sm86/matmul-with-dot-v2.py

python examples/v3.1.0/ptx_runner/sm86/matmul-with-dot-v2.py

python examples/cubin_runner/sm86/matmul-with-dot-v2.py
```

### sm75 (T4, RTX 2080, etc.)

```shell
python examples/python_runner/matmul.py

python examples/ttir_runner/matmul/matmul.py

python examples/v3.1.0/ttgir_runner/sm75/matmul.py

python examples/v3.1.0/llir_runner/sm75/matmul.py

python examples/v3.1.0/ptx_runner/sm75/matmul.py

python examples/cubin_runner/sm75/matmul.py
```