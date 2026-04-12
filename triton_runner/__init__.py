__version__ = '0.3.6'

from .version_utils import is_support_version, is_triton_geq_v3_3, triton_version
if not is_support_version:
    raise RuntimeError(f"Triton Runner doesn't support Triton v{triton_version}")


def _env_flag(name, default=True):
    import os
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "off", "no", ""}


def _init_tvm_ffi_flag():
    enabled = _env_flag("TRITON_RUNNER_ENABLE_TVM_FFI", default=False)
    if enabled:
        if not is_triton_geq_v3_3:
            raise RuntimeError(
                "TRITON_RUNNER_ENABLE_TVM_FFI requires Triton v3.3.0+ for RunnerCompiledKernel."
            )
        from .tvm_ffi import _require_tvm_ffi
        _require_tvm_ffi()
    return enabled


TRITON_RUNNER_ENABLE_TVM_FFI = _init_tvm_ffi_flag()
TRITON_RUNNER_PRODUCTION = _env_flag("TRITON_RUNNER_PRODUCTION", default=False)


from .jit import jit
from .version_utils import is_triton_geq_v3_4
if is_triton_geq_v3_4:
    from .autotune import autotune
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
