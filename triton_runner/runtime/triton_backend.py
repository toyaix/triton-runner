from __future__ import annotations

_original_triton_jit = None
_original_triton_compile = None
_original_triton_autotune = None
_original_gluon_jit = None
_env_unset = object()
_no_saved_env = object()
_original_torchinductor_cache_dir = _no_saved_env
_runner_torchinductor_cache_dir = None


def _runner_compile(src, target=None, options=None, _env_vars=None, **kwargs):
    """Route ASTSource compilation through Triton Runner's native compiler."""
    from triton.compiler.compiler import ASTSource
    from ..compiler.compile import native_compile

    if not isinstance(src, ASTSource):
        return _original_triton_compile(src, target=target, options=options, _env_vars=_env_vars, **kwargs)

    options_dict = options
    if options_dict is not None and not isinstance(options_dict, dict):
        options_dict = getattr(options_dict, "__dict__", None) or dict(options_dict)

    source_path = getattr(src.fn, "__globals__", {}).get("__file__")
    return native_compile(src, src, {}, target=target, options=options_dict, source_path=source_path)


def configure_jit_backend():
    global _original_triton_jit, _original_triton_compile, _original_gluon_jit
    global _original_torchinductor_cache_dir, _runner_torchinductor_cache_dir
    import os
    import tempfile
    import triton
    import triton.compiler.compiler as _triton_compiler
    from .. import jit as runner_jit

    if _original_triton_jit is None:
        _original_triton_jit = triton.jit
    triton.jit = runner_jit
    try:
        from triton.experimental import gluon as _triton_gluon
        from ..jit.gluon import jit as runner_gluon_jit
    except Exception:
        _triton_gluon = None
    if _triton_gluon is not None:
        if _original_gluon_jit is None:
            _original_gluon_jit = _triton_gluon.jit
        _triton_gluon.jit = runner_gluon_jit
    if _original_triton_compile is None:
        _original_triton_compile = _triton_compiler.compile
    triton.compile = _runner_compile
    _triton_compiler.compile = _runner_compile
    if _original_torchinductor_cache_dir is _no_saved_env:
        _original_torchinductor_cache_dir = os.environ.get("TORCHINDUCTOR_CACHE_DIR", _env_unset)
    if _runner_torchinductor_cache_dir is None:
        _runner_torchinductor_cache_dir = tempfile.mkdtemp(prefix="torchinductor_")
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = _runner_torchinductor_cache_dir


def restore_jit_backend():
    global _original_torchinductor_cache_dir, _runner_torchinductor_cache_dir
    import os
    import shutil
    import triton
    import triton.compiler.compiler as _triton_compiler

    if _original_triton_jit is not None:
        triton.jit = _original_triton_jit
    if _original_gluon_jit is not None:
        try:
            from triton.experimental import gluon as _triton_gluon
        except Exception:
            _triton_gluon = None
        if _triton_gluon is not None:
            _triton_gluon.jit = _original_gluon_jit
    if _original_triton_compile is not None:
        triton.compile = _original_triton_compile
        _triton_compiler.compile = _original_triton_compile
    if _original_torchinductor_cache_dir is not _no_saved_env:
        if _original_torchinductor_cache_dir is _env_unset:
            os.environ.pop("TORCHINDUCTOR_CACHE_DIR", None)
        else:
            os.environ["TORCHINDUCTOR_CACHE_DIR"] = _original_torchinductor_cache_dir
    if _runner_torchinductor_cache_dir is not None:
        shutil.rmtree(_runner_torchinductor_cache_dir, ignore_errors=True)
        _runner_torchinductor_cache_dir = None
    _original_torchinductor_cache_dir = _no_saved_env


def configure_autotune_backend():
    global _original_triton_autotune
    import triton
    from .. import autotune as runner_autotune

    if _original_triton_autotune is None:
        _original_triton_autotune = triton.autotune
    triton.autotune = runner_autotune


def restore_autotune_backend():
    import triton

    if _original_triton_autotune is not None:
        triton.autotune = _original_triton_autotune


__all__ = [
    "configure_autotune_backend",
    "configure_jit_backend",
    "restore_autotune_backend",
    "restore_jit_backend",
]
