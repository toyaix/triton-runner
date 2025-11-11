### sm90 (H100, H200, H20, etc.)
```shell
python examples/runner/v3.4.0/python/matmul-with-dot-v2.py

python examples/runner/v3.4.0/ttir/matmul-with-dot/matmul-with-dot-v2.py
```

### sm80 (A100, A30)
```shell
python examples/runner/v3.4.0/python/matmul-with-dot-v2.py

python examples/runner/v3.4.0/ttir/matmul-with-dot/matmul-with-dot-v2.py

python examples/runner/v3.4.0/cubin/sm80/matmul-with-dot-v2.py
```

### sm120 (RTX PRO 6000, RTX 5090, etc.)

**not supported**

### sm86 (A10, RTX 3090, etc.)
```shell
python examples/runner/v3.4.0/python/matmul-with-dot-v2.py

python examples/runner/v3.4.0/ttir/matmul-with-dot/matmul-with-dot-v2.py

python examples/runner/v3.1.0/ttgir/sm86/matmul-with-dot-v2.py

python examples/runner/v3.1.0/llir/sm86/matmul-with-dot-v2.py

python examples/runner/v3.1.0/ptx/sm86/matmul-with-dot-v2.py
```

### sm75 (T4, RTX 2080, etc.)

```shell
python examples/runner/v3.4.0/python/matmul.py

python examples/runner/v3.4.0/ttir/matmul/matmul.py

python examples/runner/v3.1.0/ttgir/sm75/matmul.py

python examples/runner/v3.1.0/llir/sm75/matmul.py

python examples/runner/v3.2.0/ptx/sm75/matmul.py

python examples/runner/v3.4.0/cubin/sm75/matmul.py
```