import importlib

from ..compat.version import is_triton_geq_v3_4
from .triton_backend import (
    configure_autotune_backend,
    configure_jit_backend,
    restore_autotune_backend,
    restore_jit_backend,
)

# torch helpers stay importable from this package but load on first use, so the
# core runner does not require torch to be installed. Anything that calls them
# (examples, benchmarks, dump dtype checks) still needs torch.
_TORCH_HELPER_NAMES = (
    "get_active_torch_device",
    "get_grid_dim",
    "get_n_elements_with_grid",
    "get_pad_n_elements",
    "pad_2d_to_block_shape",
)


def __getattr__(name):
    if name in _TORCH_HELPER_NAMES:
        return getattr(importlib.import_module(".torch", __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if is_triton_geq_v3_4:
    from .autotune import Autotuner, autotune

__all__ = [
    "configure_autotune_backend",
    "configure_jit_backend",
    "get_active_torch_device",
    "get_grid_dim",
    "get_n_elements_with_grid",
    "get_pad_n_elements",
    "pad_2d_to_block_shape",
    "restore_autotune_backend",
    "restore_jit_backend",
]

if is_triton_geq_v3_4:
    __all__.extend(["Autotuner", "autotune"])
