"""Demo: re-compile from specific passes across TTIR, TTGIR, and LLIR stages.

Prerequisites —— generate test data:

    export TRITON_CACHE_DIR=$PWD/.cache
    MLIR_ENABLE_DUMP=1 python examples/runner/v3.4.0/ttgir/sm90/matmul-with-tma-v4.py
    cp -r .cache/*/mlir examples/runner/pass_manage/mlir_dump/
    cp .cache/*/matmul_kernel_make_tensor_desciptor.json examples/runner/pass_manage/metadata.json

Usage:

    python examples/runner/pass_manage/demo.py
"""

import json
from pathlib import Path

import torch
import triton
import triton.language as tl
import triton_runner

HERE = Path(__file__).parent

DEVICE = triton_runner.torch_utils.get_active_torch_device()
triton_runner.configure_jit_backend()


# ---------- kernel (same as the TTGIR sm90 matmul example) ----------

@triton.jit
def matmul_kernel_make_tensor_desciptor(a_ptr, b_ptr, c_ptr,
                                        M, N, K,
                                        BLOCK_SIZE_M: tl.constexpr,
                                        BLOCK_SIZE_N: tl.constexpr,
                                        BLOCK_SIZE_K: tl.constexpr):
    pid_m = tl.program_id(axis=0)
    pid_k = tl.program_id(axis=1)

    a_desc = tl.make_tensor_descriptor(
        a_ptr, shape=[M, N], strides=[N, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
    )
    b_desc = tl.make_tensor_descriptor(
        b_ptr, shape=[N, K], strides=[K, 1],
        block_shape=[BLOCK_SIZE_N, BLOCK_SIZE_K],
    )
    c_desc = tl.make_tensor_descriptor(
        c_ptr, shape=[M, K], strides=[K, 1],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
    )
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_K), dtype=tl.float32)

    for n in range(tl.cdiv(N, BLOCK_SIZE_N)):
        a = a_desc.load([pid_m * BLOCK_SIZE_M, n * BLOCK_SIZE_N])
        b = b_desc.load([n * BLOCK_SIZE_N, pid_k * BLOCK_SIZE_K])
        accumulator = tl.dot(a, b, acc=accumulator)

    accumulator = accumulator.to(tl.float16)
    c_desc.store([pid_m * BLOCK_SIZE_M, pid_k * BLOCK_SIZE_K], accumulator)


def run_matmul(a, b, **extra_kwargs):
    M, N = a.shape
    N2, K = b.shape
    assert N == N2

    c = torch.empty((M, K), device=a.device, dtype=torch.float16)

    def alloc_fn(size, alignment, stream):
        return torch.empty(size, device="cuda", dtype=torch.int8)
    triton.set_allocator(alloc_fn)

    def grid(META):
        return (
            triton.cdiv(M, META['BLOCK_SIZE_M']),
            triton.cdiv(K, META['BLOCK_SIZE_K']),
        )

    matmul_kernel_make_tensor_desciptor[grid](
        a, b, c, M, N2, K,
        BLOCK_SIZE_M=128, BLOCK_SIZE_K=64, BLOCK_SIZE_N=64,
        **extra_kwargs,
    )
    return c


# ---------- locate test data ----------

MLIR_DUMP = HERE / "mlir_dump"
METADATA = json.loads((HERE / "metadata.json").read_text())


# ---------- 10 pass test cases across 3 stages ----------

CASES = [
    # --- TTIR stage (3 passes) ---
    ("TTIR / rewrite_tensor_ptr",   "ttir_src",  "rewrite_tensor_pointer",   "07-TritonRewriteTensorPointer.mlir"),
    ("TTIR / combine_ops",          "ttir_src",  "combine_ops",              "09-TritonCombineOps.mlir"),
    ("TTIR / loop_unroll",          "ttir_src",  "loop_unroll",              "13-TritonLoopUnroll.mlir"),

    # --- TTGIR stage (5 passes) ---
    ("TTGIR / coalesce",            "ttgir_src", "coalesce",                 "15-TritonGPUCoalesce.mlir"),
    ("TTGIR / remove_layout_conv",  "ttgir_src", "remove_layout_conversions","18-changed-TritonGPURemoveLayoutConversions.mlir"),
    ("TTGIR / accelerate_matmul",   "ttgir_src", "accelerate_matmul",        "20-changed-TritonGPUAccelerateMatmul.mlir"),
    ("TTGIR / fuse_nested_loops",   "ttgir_src", "fuse_nested_loops",        "26-TritonGPUFuseNestedLoops.mlir"),
    ("TTGIR / pipeline",            "ttgir_src", "pipeline",                 "43-changed-TritonGPUPipeline.mlir"),

    # --- LLIR stage (2 passes, require metadata_json) ---
    ("LLIR / allocate_shared_mem",  "llir_src",  "allocate_shared_memory",   "67-changed-AllocateSharedMemory.mlir"),
    ("LLIR / allocate_warp_groups", "llir_src",  "allocate_warp_groups",     "65-changed-TritonGPUAllocateWarpGroups.mlir"),
]

# ---------- run ----------

M, N, K = 1024, 512, 256
a = torch.randn((M, N), device=DEVICE, dtype=torch.float16).to(torch.float8_e5m2)
b = torch.randn((N, K), device=DEVICE, dtype=torch.float16).to(torch.float8_e5m2)

torch_output = torch.matmul(a.to(torch.float16), b.to(torch.float16))

print(f"{'Test Case':<38s} {'Match':>6s}  Result")
print("-" * 95)

passed = 0
failed = 0

for label, source_kwarg, start_pass, filename in CASES:
    src_file = str(MLIR_DUMP / filename)
    kwargs = {
        source_kwarg: src_file,
        "start_pass": start_pass,
    }
    if source_kwarg == "llir_src":
        kwargs["metadata_json"] = METADATA

    try:
        result = run_matmul(a, b, **kwargs)
        ok = torch.allclose(result.float(), torch_output.float(), atol=0.125, rtol=0)
    except Exception as e:
        ok = False
        result = str(e)[:80]

    status = "OK" if ok else "FAIL"
    print(f"{label:<38s} {status:>6s}  {result if not ok else ''}")

    if ok:
        passed += 1
    else:
        failed += 1

print("-" * 95)
print(f"Passed: {passed}/{len(CASES)}, Failed: {failed}/{len(CASES)}")
