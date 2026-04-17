from ..compat.version import is_triton_geq_v3_4
from .triton_backend import (
    configure_autotune_backend,
    configure_jit_backend,
    restore_autotune_backend,
    restore_jit_backend,
)
from .torch import (
    get_active_torch_device,
    get_grid_dim,
    get_n_elements_with_grid,
    get_pad_n_elements,
    pad_2d_to_block_shape,
)

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
