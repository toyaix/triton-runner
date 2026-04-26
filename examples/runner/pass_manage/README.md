# Pass Management Examples

Control which compilation pass Triton Runner starts from using `start_pass`.

## Supported Versions

| Triton Version | Status |
|---|---|
| v3.4.0 | Supported |
| v3.7.0 | Planned |

Pass pipelines are **version-specific and architecture-sensitive** — each Triton version ships different pass definitions, and pass ordering also depends on SM capability (`capability // 10`).

## Prerequisites

Generate the test data and copy it into this directory:

```shell
export TRITON_CACHE_DIR=$PWD/.cache
MLIR_ENABLE_DUMP=1 python examples/runner/v3.4.0/ttgir/sm90/matmul-with-tma-v4.py
cp -r .cache/*/mlir examples/runner/pass_manage/mlir_dump/
cp .cache/*/matmul_kernel_make_tensor_desciptor.json examples/runner/pass_manage/metadata.json
```

`TRITON_CACHE_DIR` is optional but keeps the cache predictable.

## Usage

```shell
python examples/runner/pass_manage/demo.py
```

`demo.py` reads the local `mlir_dump/` and `metadata.json`. It tests 10 passes across three stages and verifies each result against a Torch reference:

| Stage | Passes | Source Kwarg |
|---|---|---|
| TTIR | rewrite_tensor_pointer, combine_ops, loop_unroll | `ttir_src` |
| TTGIR | coalesce, remove_layout_conversions, accelerate_matmul, fuse_nested_loops, pipeline | `ttgir_src` |
| LLIR | allocate_shared_memory, allocate_warp_groups | `llir_src` + `metadata_json` |

## API

```python
result = your_kernel[grid](
    a, b, c, *args,
    ttgir_src="path/to/18-changed-TritonGPURemoveLayoutConversions.mlir",
    start_pass="remove_layout_conversions",
)
```

- `ttir_src` / `ttgir_src` — path to a `.mlir` dump file, no metadata required
- `llir_src` — path to a `.mlir` dump file, **requires `metadata_json`** (contains `name`, `shared`, `num_warps`, etc.)
- `start_pass` — short name matching a pass in the pipeline (see `pass_pipeline.py` for the full list)

LLIR passes must be **before** the `to_llvmir` pass.

## How it works

```
Normal compilation:   [TTIR passes...] → [TTGIR passes...] → [LLIR passes...] → LLVM conv → [PTX] → [cubin]

With start_pass:                       → [remaining passes in stage] → ...
                                          start here
```

The `start_pass` parameter identifies which pass the source file corresponds to,
so the runner can skip already-executed passes and only run the remaining ones.
