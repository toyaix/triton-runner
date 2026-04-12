__version__ = '0.3.6'

from .version_utils import is_support_version, is_triton_v3_4, triton_version
if not is_support_version:
    raise RuntimeError(f"Triton Runner doesn't support Triton v{triton_version}")


def _env_flag(name, default=True):
    import os
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "off", "no", ""}


def _init_is_cuda():
    from triton.runtime import driver
    from triton.backends.compiler import GPUTarget
    target = driver.active.get_current_target()
    return isinstance(target, GPUTarget) and target.backend == "cuda"


IS_CUDA = _init_is_cuda()

TRITON_RUNNER_PROD = _env_flag("TRITON_RUNNER_PROD", default=False)


from .version_utils import is_triton_geq_v3_4
if is_triton_geq_v3_4:
    from .autotune import autotune

if TRITON_RUNNER_PROD and IS_CUDA and is_triton_v3_4:
    from .tvm_ffi import _require_tvm_ffi
    _require_tvm_ffi()
    from .jit_prod import jit
    from .color_print import blue_print
    blue_print("[Triton Runner] Production mode enabled")
else:
    from .jit import jit
from . import color_print
from . import torch_utils

_original_triton_jit = None


def configure_jit_backend():
    global _original_triton_jit
    import triton
    if _original_triton_jit is None:
        _original_triton_jit = triton.jit
    triton.jit = jit


def restore_jit_backend():
    import triton
    if _original_triton_jit is not None:
        triton.jit = _original_triton_jit


_original_triton_autotune = None


def configure_autotune_backend():
    global _original_triton_autotune
    import triton
    if _original_triton_autotune is None:
        _original_triton_autotune = triton.autotune
    triton.autotune = autotune


def restore_autotune_backend():
    import triton
    if _original_triton_autotune is not None:
        triton.autotune = _original_triton_autotune


def get_file_dir(file):
    import os
    return os.path.dirname(os.path.abspath(file))
