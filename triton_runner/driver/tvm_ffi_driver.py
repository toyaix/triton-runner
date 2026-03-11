"""TVM-FFI based CUDA kernel launcher following the driver/v3_5_0 pattern."""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from typing import Any

_module_cache: dict[str, Any] = {}
_cache_lock = threading.Lock()


def _get_tvm_ffi_cache_dir() -> str:
    """Return the on-disk directory used to persist compiled TVM-FFI modules."""
    cache_dir = os.environ.get("TRITON_CACHE_DIR")
    if cache_dir is None:
        cache_dir = os.path.join(Path.home(), ".triton", "cache")
    return os.path.join(cache_dir, "tvm_ffi_launcher")


class TvmFfiLauncher:
    """TVM-FFI kernel launcher with the same call convention as ``CudaLauncher``."""

    def __init__(self, src, metadata, asm):
        from ..tvm_ffi import (
            _CompiledArtifact,
            _module_name_for_metadata,
            _normalize_metadata,
            _parse_kernel_signature,
            _render_cuda_shim,
            _require_tvm_ffi,
            _sanitize_identifier,
        )

        metadata_dict = _normalize_metadata(metadata)

        cubin_bytes = asm.get("cubin") if isinstance(asm, dict) else getattr(asm, "get", lambda k: None)("cubin")
        if cubin_bytes is None:
            raise ValueError("TVM-FFI launcher requires asm['cubin'].")
        if isinstance(cubin_bytes, memoryview):
            cubin_bytes = cubin_bytes.tobytes()
        elif isinstance(cubin_bytes, bytearray):
            cubin_bytes = bytes(cubin_bytes)

        kernel_signature = metadata_dict.get("kernel_signature")
        signature = _parse_kernel_signature(kernel_signature)
        self._is_constexpr = tuple(entry.is_constexpr for entry in signature)

        artifact = _CompiledArtifact(
            kernel_name=str(metadata_dict["name"]),
            module_name=_module_name_for_metadata(metadata_dict),
            cubin_bytes=cubin_bytes,
            metadata=metadata_dict,
            signature=signature,
        )

        cuda_source = _render_cuda_shim(artifact)
        cache_hash = hashlib.sha256(cuda_source.encode("utf-8") + cubin_bytes).hexdigest()

        with _cache_lock:
            cached = _module_cache.get(cache_hash)

        if cached is not None:
            self._tvm_mod = cached
        else:
            _, cpp = _require_tvm_ffi()

            build_dir = os.path.join(_get_tvm_ffi_cache_dir(), cache_hash)
            os.makedirs(build_dir, exist_ok=True)

            embed_name = _sanitize_identifier(artifact.module_name)

            mod = cpp.load_inline(
                artifact.module_name,
                cuda_sources=cuda_source,
                build_directory=build_dir,
                embed_cubin={embed_name: cubin_bytes},
                keep_module_alive=True,
                extra_cuda_cflags=[],
                extra_ldflags=["-lcudart", "-lcuda"],
            )

            self._tvm_mod = mod
            with _cache_lock:
                _module_cache[cache_hash] = mod

        self._kernel_name = artifact.kernel_name
        self._tvm_func = self._tvm_mod[self._kernel_name]

    def __call__(self, gridX, gridY, gridZ, stream, function, *args):
        packed_metadata = args[0]
        launch_metadata = args[1]
        launch_enter_hook = args[2]
        launch_exit_hook = args[3]
        bound_args = args[4:]

        if launch_enter_hook is not None:
            launch_enter_hook(launch_metadata)
        non_constexpr_args = [
            arg for arg, is_const in zip(bound_args, self._is_constexpr)
            if not is_const
        ]
        self._tvm_func(gridX, gridY, gridZ, *non_constexpr_args)

        if launch_exit_hook is not None:
            launch_exit_hook(launch_metadata)
