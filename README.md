# triton_runner

triton_runner(Triton multi-level runner) is a lightweight, multi-level execution engine for [Triton](https://github.com/triton-lang/triton), designed to support IR/PTX/cubin launches in complex pass pipelines.

This project is built specifically for **Triton v3.3.x** and is not guaranteed to work with other versions.

## Example

> **Note:** The following example requires an NVIDIA GPU with compute capability `sm90 (H100, H200, H20, etc.)`, `sm80 (A100, A30)`, `sm120 (RTX PRO 6000, RTX 5090, etc.)`, `sm86 (A10, RTX 3090, etc.)` or `sm75 (T4, RTX 2080, etc.)`. Please make sure to install the package before running the example.

> If your GPU does not have one of the above compute capabilities, you can use `TRITON_CACHE_DIR=$PWD/.cache` to output the Triton cache to the current directory, and then copy the corresponding cache files to your target machine.

Here is an example command for `sm90`. For more examples, please refer to [examples](./doc/examples.md).

### sm90 (H100, H200, H20, etc.)
```bash
python examples/ttir_runner/matmul.py

python examples/ttgir_runner/sm90/matmul-with-tma-v3.py

python examples/llir_runner/sm90/matmul-with-tma-v3.py

python examples/ptx_runner/sm90/matmul-with-tma-v3.py

python examples/cubin_runner/sm90/matmul-with-tma-v3.py
```

## Installation

install the package as a standard Python package

```bash
git clone https://github.com/OpenMLIR/triton_runner
cd triton_runner
pip install .
```

### Development Installation (Editable Mode)

If you are actively developing or modifying the source code, install the package in editable mode. This allows changes in the source files to take effect immediately without reinstalling:

```bash
pip install -e .
```

## ⚠️ Version Compatibility

This runner is built against **Triton v3.3.x**.
Compatibility with other versions of Triton is **not guaranteed** and may lead to unexpected behavior or run failures.

## 📄 License

This project is licensed under the **MIT License**.
See the [LICENSE](./LICENSE) file for more details.
