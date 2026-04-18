---
name: triton-generate-artifacts
description: Generate Triton Runner ttir, ttgir, llir, ptx, cubin, and metadata artifacts inside examples/runner/vX.Y.Z for the active Triton version and the current machine's GPU capability. Use when publishing a new Triton version's runner artifacts, regenerating checked-in example kernels, or producing compile-only artifacts for extra NVIDIA capabilities on another machine.
---

# Triton Generate Artifacts

Use this skill when the task is to generate or refresh versioned runner artifacts under `examples/runner/vX.Y.Z`.

## Workflow

1. Confirm the active Triton version with `python -c "import triton; print(triton.__version__)"`.
2. Use the bundled generator script in this skill as the source of truth.
3. Default to the current machine's GPU capability only.
4. Generate extra capabilities only when the user explicitly asks for them.
5. Keep README commands pointing at checked-in static artifacts; use this skill only to produce or refresh those files.

## Commands

From the repository root:

```bash
python skills/triton-generate-artifacts/scripts/generate_runner_artifacts.py
```

To target explicit capabilities:

```bash
python skills/triton-generate-artifacts/scripts/generate_runner_artifacts.py 75 80 86 90 120
```

To target another Triton version directory without changing the environment:

```bash
python skills/triton-generate-artifacts/scripts/generate_runner_artifacts.py --version 3.7.0 75
```

## Validation

- Verify the target version directory contains `ttir`, `ttgir`, `llir`, `ptx`, and `cubin` outputs plus metadata json where applicable.
- For the local machine capability, run the corresponding README example commands when GPU execution is available.
- For extra capabilities generated via compile-only override, validate file presence and metadata `arch` and `triton_version` instead of runtime execution.
