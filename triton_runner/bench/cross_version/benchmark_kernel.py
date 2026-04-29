"""BenchmarkKernel — the contract a kernel module must fulfill for cross-version benchmarking.

A kernel module is a ``.py`` file that exposes three required names and one optional one.
The runner uses ``importlib`` to load it inside each conda environment, so the module must
be self-contained (no relative sibling imports that wouldn't resolve from a different CWD).

Required
--------
``kernel`` : jit-decorated callable
    The Triton kernel function.  Usually ``kernel = triton.jit(my_kernel)``.
    The runner calls ``mod.kernel[grid](*args)``.

``prepare_args(**size) -> tuple``
    Accepts keyword arguments matching the problem-space dimension names (e.g. ``M=1024, N=1024, K=1024``)
    and returns a tuple of concrete tensors / scalars ready for the kernel launch.

``get_grid(**size) -> tuple``
    Same keyword arguments; returns a ``(grid_x, grid_y, grid_z)`` launch-grid tuple.

Optional
--------
``get_problem_space() -> ProblemSpace | dict | None``
    Returns the benchmark input space for auto-discovery.
    If omitted, the user must pass sizes via CLI or Python API.

Example (matmul) ::

    # kernels/matmul.py
    import torch, triton, triton.language as tl
    from triton_runner.bench.cross_version import ProblemSpace

    @triton.jit
    def matmul_kernel(a_ptr, b_ptr, c_ptr, M, N, K, ...):
        ...

    def prepare_args(M, N, K):
        a = torch.randn((M, K), device="cuda", dtype=torch.float16)
        b = torch.randn((K, N), device="cuda", dtype=torch.float16)
        c = torch.empty((M, N), device="cuda", dtype=torch.float16)
        return (a, b, c)

    def get_grid(M, N, K):
        return (triton.cdiv(N, 128), triton.cdiv(M, 128), 1)

    def get_problem_space():
        return ProblemSpace.matmul_square([256, 512, 1024, 2048, 4096])

    kernel = matmul_kernel
"""

from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable

from .problem_space import ProblemSpace


@runtime_checkable
class BenchmarkKernel(Protocol):
    """Protocol that a cross-version-benchmarkable kernel module satisfies.

    This is a structural protocol — any module exposing the following names
    matches, no inheritance required.
    """

    kernel: Callable[..., Any]
    """The jit-decorated Triton kernel. Called as ``kernel[grid](*args)``."""

    def prepare_args(self, **size: Any) -> tuple:
        """Build input tensors from problem-size kwargs.

        The kwargs keys correspond to the ProblemSpace dimension names.
        Must return a tuple suitable for ``kernel[grid](*args)``.
        """
        ...

    def get_grid(self, **size: Any) -> tuple[int, int, int]:
        """Return a ``(grid_x, grid_y, grid_z)`` launch grid for the given size."""
        ...

    def get_kernel_kwargs(self, **size: Any) -> dict:
        """Optional: return constexpr / launch kwargs for ``kernel[grid](*args, **kwargs)``.

        Use for kernels with ``tl.constexpr`` parameters or custom ``num_stages`` / ``num_warps``.
        """
        ...

    def get_problem_space(self) -> ProblemSpace | dict | None:
        """Optionally return the benchmark input space for auto-discovery."""
        ...


# ---- helper: validate a loaded kernel module ----

def validate_kernel_module(mod, kernel_path: str) -> list[str]:
    """Check that a loaded module satisfies the BenchmarkKernel contract.

    Returns a list of human-readable issue descriptions (empty = valid).
    """
    issues: list[str] = []

    for attr in ("kernel", "prepare_args", "get_grid"):
        if not hasattr(mod, attr):
            issues.append(f"Missing required attribute: '{attr}'")
        elif not callable(getattr(mod, attr)):
            issues.append(f"'{attr}' must be callable")

    if issues:
        issues.insert(0, f"Kernel module {kernel_path!r} does not satisfy the BenchmarkKernel contract:")

    return issues
