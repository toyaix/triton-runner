__version__ = '0.3.7'

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

TRITON_RUNNER_PROD_TEST = _env_flag("TRITON_RUNNER_PROD_TEST", default=False)
TRITON_RUNNER_PROD = _env_flag("TRITON_RUNNER_PROD", default=False) or TRITON_RUNNER_PROD_TEST


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
_original_triton_compile = None


def _runner_compile(src, target=None, options=None, _env_vars=None, **kwargs):
    """Drop-in replacement for ``triton.compile`` that routes through
    :func:`triton_runner.compile.native_compile` so TorchInductor-generated
    kernels (which call ``triton.compile(ASTSource(...), ...)`` directly
    and bypass the ``@triton.jit`` decorator path) still flow through the
    Triton Runner pipeline.
    """
    from triton.compiler.compiler import ASTSource
    from .compile import native_compile

    if not isinstance(src, ASTSource):
        # IRSource / string paths are not produced by Inductor; fall back
        # to the stock compiler so we don't have to synthesise an ast_src.
        return _original_triton_compile(src, target=target, options=options, _env_vars=_env_vars, **kwargs)

    options_dict = options
    if options_dict is not None and not isinstance(options_dict, dict):
        options_dict = getattr(options_dict, "__dict__", None) or dict(options_dict)

    return native_compile(src, src, {}, target=target, options=options_dict)


def configure_jit_backend():
    global _original_triton_jit, _original_triton_compile
    import triton
    import triton.compiler.compiler as _triton_compiler
    if _original_triton_jit is None:
        _original_triton_jit = triton.jit
    triton.jit = jit
    if _original_triton_compile is None:
        _original_triton_compile = _triton_compiler.compile
    triton.compile = _runner_compile
    _triton_compiler.compile = _runner_compile


def restore_jit_backend():
    import triton
    import triton.compiler.compiler as _triton_compiler
    if _original_triton_jit is not None:
        triton.jit = _original_triton_jit
    if _original_triton_compile is not None:
        triton.compile = _original_triton_compile
        _triton_compiler.compile = _original_triton_compile


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
