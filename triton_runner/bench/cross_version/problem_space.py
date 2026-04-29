"""ProblemSpace — defines the benchmark input space for a kernel.

Each kernel type has different "keys" (dimensions) and ranges.  This module
provides a unified way to declare, generate, and load those input configurations.

Three generation modes::

    # zip — pairwise alignment (default, for "square" sweeps like matmul)
    ps = ProblemSpace(
        dimensions={"M": [256, 512, 1024], "N": [256, 512, 1024], "K": [256, 512, 1024]},
        mode="zip",
    )
    ps.generate()  # → [{"M": 256, "N": 256, "K": 256}, {"M": 512, "N": 512, "K": 512}, ...]

    # product — cartesian product (exhaustive)
    ps = ProblemSpace(
        dimensions={"M": [256, 1024], "N": [256, 1024], "K": [1024]},
        mode="product",
    )
    ps.generate()  # → 2 x 2 x 1 = 4 combos

    # sweep — vary one dimension, fix the rest
    ps = ProblemSpace(
        dimensions={"M": [256, 512, 1024, 2048, 4096, 8192]},
        mode="sweep",
        sweep_dim="M",
        fixed={"N": 1024, "K": 1024},
    )
    ps.generate()  # → [{"M": 256, "N": 1024, "K": 1024}, {"M": 512, ...}, ...]
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence


@dataclass
class ProblemSpace:
    """Declarative benchmark input space.

    Attrs:
        dimensions: mapping from dimension name to a list of values.
        mode: ``"zip"`` (pairwise), ``"product"`` (cartesian), or ``"sweep"``.
        sweep_dim: the dimension being swept (required for ``mode="sweep"``).
        fixed: fixed values for non-sweep dimensions (required for ``mode="sweep"``).
    """

    dimensions: dict[str, list[Any]]
    mode: str = "zip"
    sweep_dim: str | None = None
    fixed: dict[str, Any] | None = None

    def generate(self) -> list[dict[str, Any]]:
        """Expand the problem space into a flat list of size dicts."""
        dims = self.dimensions

        if self.mode == "zip":
            # all dimension lists must be same length
            keys = list(dims.keys())
            length = len(dims[keys[0]])
            if any(len(dims[k]) != length for k in keys):
                raise ValueError(
                    f"All dimension lists must be the same length in zip mode. "
                    f"Got: {{{', '.join(f'{k}: {len(v)}' for k, v in dims.items())}}}"
                )
            return [dict(zip(keys, values)) for values in zip(*dims.values())]

        if self.mode == "product":
            keys = list(dims.keys())
            return [
                dict(zip(keys, combo))
                for combo in itertools.product(*dims.values())
            ]

        if self.mode == "sweep":
            if self.sweep_dim is None:
                raise ValueError("sweep_dim is required for mode='sweep'")
            if self.fixed is None:
                raise ValueError("fixed is required for mode='sweep'")
            sweep_values = dims[self.sweep_dim]
            return [{**self.fixed, self.sweep_dim: v} for v in sweep_values]

        raise ValueError(f"Unknown mode: {self.mode!r}. Expected 'zip', 'product', or 'sweep'.")

    @property
    def size(self) -> int:
        """Number of problem configurations this space expands to."""
        return len(self.generate())

    # ---- factories ----

    @classmethod
    def from_kernel(cls, kernel_path: str) -> ProblemSpace | None:
        """Try to load ``get_problem_space()`` from a kernel module.

        The kernel module may optionally define::

            def get_problem_space():
                return ProblemSpace(
                    dimensions={"M": [256, 512, 1024, 2048], ...},
                    mode="zip",
                )
        """
        import importlib.util

        path = Path(kernel_path).resolve()
        if not path.exists():
            return None

        try:
            spec = importlib.util.spec_from_file_location(
                "_ps_kernel", str(path),
            )
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
        except Exception:
            return None

        if hasattr(mod, "get_problem_space"):
            ps = mod.get_problem_space()
            if isinstance(ps, cls):
                return ps
            # also accept plain dict → ProblemSpace
            if isinstance(ps, dict):
                return cls(**ps)

        return None

    @classmethod
    def from_json(cls, path: str) -> ProblemSpace | None:
        """Load a ProblemSpace from a JSON file.

        JSON format::

            {
                "dimensions": {"M": [256, 512, 1024], "N": [256, 512, 1024], "K": [256, 512, 1024]},
                "mode": "zip"
            }

        Or, a plain list of size dicts (backward-compatible)::

            [{"M": 256, "N": 256, "K": 256}, {"M": 512, "N": 512, "K": 512}]
        """
        import json

        data = json.loads(Path(path).read_text())

        if isinstance(data, list):
            # backward-compatible: plain list of size dicts
            if not data:
                return None
            dims = {}
            for entry in data:
                for k, v in entry.items():
                    dims.setdefault(k, []).append(v)
            return cls(dimensions=dims, mode="zip")

        if isinstance(data, dict) and "dimensions" in data:
            return cls(**data)

        return None

    # ---- auto-generate helpers ----

    @classmethod
    def matmul_square(cls, sizes: Sequence[int] | None = None) -> ProblemSpace:
        """Convenience: square matmul (M = N = K) over typical power-of-two sizes."""
        sizes = list(sizes) if sizes else [256, 512, 1024, 2048, 4096, 8192]
        return cls(
            dimensions={"M": sizes, "N": sizes, "K": sizes},
            mode="zip",
        )

    @classmethod
    def matmul_sweep_m(cls, fixed_n: int = 1024, fixed_k: int = 1024,
                       sizes: Sequence[int] | None = None) -> ProblemSpace:
        """Convenience: sweep M while fixing N, K."""
        sizes = list(sizes) if sizes else [128, 256, 512, 1024, 2048, 4096, 8192]
        return cls(
            dimensions={"M": sizes},
            mode="sweep",
            sweep_dim="M",
            fixed={"N": fixed_n, "K": fixed_k},
        )

    @classmethod
    def sweep(cls, dim: str, values: Sequence,
              fixed: dict[str, Any]) -> ProblemSpace:
        """Generic single-dimension sweep."""
        return cls(
            dimensions={dim: list(values)},
            mode="sweep",
            sweep_dim=dim,
            fixed=fixed,
        )
