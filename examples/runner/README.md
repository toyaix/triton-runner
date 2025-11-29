Example commands for the multi-level runner with Triton **v3.4.0**. For other Triton versions, see the corresponding examples directory.
- For Triton v3.3.1 or v3.3.0, in [v3.3.x](./v3.3.x).
- For Triton v3.2.0, in [v3.2.0](./v3.2.0).
- For Triton v3.1.0, in [v3.1.0](./v3.1.0).
- For Triton v3.0.0, in [v3.0.0](./v3.0.0).


### sm90 (H100, H200, H20, etc.)
```shell
python examples/runner/v3.5.x/python/matmul-with-tma-v4.py

python examples/runner/v3.5.x/ttir/matmul-with-tma/matmul-with-tma-v4.py

python examples/runner/v3.4.0/ttgir/sm90/matmul-with-tma-v4.py

python examples/runner/v3.4.0/llir/sm90/matmul-with-tma-v4.py

python examples/runner/v3.4.0/ptx/sm90/matmul-with-tma-v4.py

python examples/runner/v3.4.0/cubin/sm90/matmul-with-tma-v4.py

python examples/runner/v3.5.x/gluon/01-intro.py
python examples/runner/v3.5.x/gluon/02-layouts.py
```

### sm80 (A100, A30)
```shell
python examples/runner/v3.5.x/python/matmul-with-dot-v2.py

python examples/runner/v3.5.x/ttir/matmul-with-dot/matmul-with-dot-v2.py

python examples/runner/v3.4.0/ttgir/sm80/matmul-with-dot-v2.py

python examples/runner/v3.4.0/llir/sm80/matmul-with-dot-v2.py

python examples/runner/v3.4.0/ptx/sm80/matmul-with-dot-v2.py

python examples/runner/v3.4.0/cubin/sm80/matmul-with-dot-v2.py

python examples/runner/v3.5.x/gluon/01-intro.py
python examples/runner/v3.5.x/gluon/02-layouts.py
```

### sm120 (RTX PRO 6000, RTX 5090, etc.)
```shell
python examples/runner/v3.5.x/python/matmul-with-tma-v4.py

python examples/runner/v3.5.x/ttir/matmul-with-tma/matmul-with-tma-v4.py

python examples/runner/v3.4.0/ttgir/sm120/matmul-with-tma-v4.py

python examples/runner/v3.4.0/llir/sm120/matmul-with-tma-v4.py

python examples/runner/v3.4.0/ptx/sm120/matmul-with-tma-v4.py

python examples/runner/v3.4.0/cubin/sm120/matmul-with-tma-v4.py

python examples/runner/v3.5.x/gluon/01-intro.py
python examples/runner/v3.5.x/gluon/02-layouts.py
```

### sm86 (A10, RTX 3090, etc.)
```shell
python examples/runner/v3.5.x/python/matmul-with-dot-v2.py

python examples/runner/v3.5.x/ttir/matmul-with-dot/matmul-with-dot-v2.py

python examples/runner/v3.4.0/ttgir/sm86/matmul-with-dot-v2.py

python examples/runner/v3.4.0/llir/sm86/matmul-with-dot-v2.py

python examples/runner/v3.4.0/ptx/sm86/matmul-with-dot-v2.py

python examples/runner/v3.4.0/cubin/sm86/matmul-with-dot-v2.py

python examples/runner/v3.5.x/gluon/01-intro.py
python examples/runner/v3.5.x/gluon/02-layouts.py
```

### sm75 (T4, RTX 2080, etc.)
```shell
python examples/runner/v3.5.x/python/matmul.py

python examples/runner/v3.5.x/ttir/matmul/matmul.py

python examples/runner/v3.5.x/ttgir/sm75/matmul.py

python examples/runner/v3.4.0/llir/sm75/matmul.py

python examples/runner/v3.4.0/ptx/sm75/matmul.py

python examples/runner/v3.4.0/cubin/sm75/matmul.py

python examples/runner/v3.5.x/gluon/01-intro.py
python examples/runner/v3.5.x/gluon/02-layouts.py
```
