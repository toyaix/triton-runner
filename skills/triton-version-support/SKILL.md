---
name: triton-version-support
description: Add, fix, or validate Triton Runner support for an exact Triton version. Use when the user names a specific Triton release such as 3.7.0 or 3.8.0 and wants Codex to adapt compatibility gates, JIT shims, example docs, or regression coverage in the Triton Runner project, then verify the result with the matching Triton install and CUDA-backed tests.
---

# Triton Version Support

Adapt Triton Runner to the requested Triton version with the smallest defensible diff. Treat the installed Triton package as the source of truth for runtime and JIT behavior, and validate with real CUDA-backed commands before concluding the work is done.

## Workflow

### 1. Confirm the exact target version

Read the user's requested Triton version as an exact value such as `3.7.0`.

Check the active environment first:

```bash
python -c "import triton; print(triton.__version__)"
```

If the active version does not match and the user expects end-to-end validation on that exact version, switch or install the requested version before making claims about compatibility. Use escalated execution when `pip install` or CUDA access is required.

### 2. Inspect the version-sensitive surfaces

Start by reading the files that usually gate Triton version support in this repository:

- `triton_runner/compat/version.py`
- `triton_runner/compat/__init__.py`
- `triton_runner/jit/versions.py`
- `triton_runner/jit/api.py`
- `triton_runner/jit/gluon.py`
- `triton_runner/__init__.py`
- `test/test.py`
- `test/regression_test.py`
- `examples/runner/v*/README.md`

Inspect the installed Triton implementation instead of guessing API details. Prefer direct introspection:

```bash
python - <<'PY'
import inspect
from triton.runtime.jit import JITFunction
print(inspect.getsource(JITFunction))
PY
```

Check `compute_cache_key`, `JITFunction.run`, `JITFunction._do_compile`, binder layout, hook signatures, and any changed launch arguments before deciding what to override.

### 3. Implement the minimal version-specific change

Prefer the smallest change that matches the actual Triton delta:

- Add a new `RunnerJITFunctionVx_y_z` class only when the requested version materially differs from an existing one.
- Override only `run()` when that is sufficient.
- Reuse inherited helper methods when they still match the installed Triton contract.
- Keep the code style aligned with nearby version-specific classes.
- Update dispatch tables and support booleans after the JIT path is correct.

When extending version coverage, update the usual outer layers:

- Support range and booleans in `triton_runner/compat/version.py`
- Re-exports in `triton_runner/compat/__init__.py`
- JIT dispatch in `triton_runner/jit/api.py`
- Gluon dispatch in `triton_runner/jit/gluon.py` if applicable
- Example directory selection via `uni_triton_version`
- Regression matrix defaults in `test/regression_test.py` when the new version should be part of the default sweep

### 4. Keep examples and docs in sync

If `test/test.py` loads commands from `examples/runner/v{uni_triton_version}/README.md`, create the matching example folder when introducing a new exact version. If the new release behaves like the previous one, copy the prior README first and then adjust only if needed.

Typical command:

```bash
python examples/runner/python/triton/matmul.py
```

### 5. Validate in escalating steps

Run fast checks before long regressions:

```bash
python -m py_compile triton_runner/jit/versions.py triton_runner/jit/api.py triton_runner/compat/version.py
python examples/runner/python/triton/matmul.py
python test/regression_test.py 3.7.0
```

Use escalated execution for commands that need CUDA visibility or package installation. Distinguish environment warnings from real compatibility failures; do not report unrelated `pip` warnings as code regressions.

### 6. Report with concrete outcomes

In the final summary:

- State which Triton version was validated.
- Name the key compatibility files changed.
- State whether `matmul.py` passed.
- State whether the targeted regression command passed.
- Call out residual risks only when a test could not be run or a subsystem was not exercised.

## Heuristics

- Prefer primary evidence from the installed Triton package over memory.
- Prefer minimal diffs over broad refactors.
- Prefer exact version names such as `3.7.0` in examples and reports.
- Prefer fixing the real failing surface instead of preemptively rewriting unrelated compatibility code.
