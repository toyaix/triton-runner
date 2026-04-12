"""TVM-FFI based CUDA kernel launcher following the driver/v3_5_0 pattern."""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import Any

from ._cuda_build import build_module_from_src, cuda_include_dirs, library_dirs

_GENERIC_LAUNCHER_NAME = "triton_runner_tvm_ffi_generic_launcher"
_generic_module_cache: dict[str, Any] = {}
_registered_kernel_cache: dict[tuple[str, str], int] = {}
_cache_lock = threading.Lock()
_GENERIC_LAUNCHER_LIBRARIES = ["tvm_ffi", "cudart", "cuda", "dl"]
_ARG_KIND_CODES = {
    "pointer": 0,
    "i1": 1,
    "i8": 2,
    "i16": 3,
    "i32": 4,
    "i64": 5,
    "u1": 6,
    "u8": 7,
    "u16": 8,
    "u32": 9,
    "u64": 10,
    "fp16": 11,
    "bf16": 12,
    "fp32": 13,
    "f32": 13,
    "fp64": 14,
    "tensordesc": 15,
}


def _get_generic_launcher_source_path() -> Path:
    return Path(__file__).with_name("generic_launcher.cc")


def _build_generic_launcher_module(tvm_ffi_module: Any, module_name: str, build_dir: str | Path, source: str) -> Any:
    from . import _shared_library_path

    tvm_ffi_root = Path(tvm_ffi_module.__file__).resolve().parent
    include_dirs = [str(tvm_ffi_root / "include"), *cuda_include_dirs()]
    return build_module_from_src(
        module_name=module_name,
        build_dir=build_dir,
        source=source,
        include_dirs=include_dirs,
        library_dirs=library_dirs(tvm_ffi_root / "lib", include_stubs=True),
        libraries=_GENERIC_LAUNCHER_LIBRARIES,
        ccflags=("-std=c++17",),
        source_ext=".cc",
        final_path=_shared_library_path(build_dir, module_name),
        load_module=lambda path: tvm_ffi_module.load_module(path, keep_module_alive=True),
    )


def _registration_token_for_payload(payload: tuple[Any, ...]) -> str:
    hasher = hashlib.sha256()

    def _update(value: Any) -> None:
        if isinstance(value, str):
            hasher.update(b"s\0")
            hasher.update(value.encode("utf-8"))
            hasher.update(b"\0")
            return
        if isinstance(value, int):
            hasher.update(f"i{value}\0".encode("ascii"))
            return
        if isinstance(value, (bytes, bytearray, memoryview)):
            raw = bytes(value)
            hasher.update(b"b\0")
            hasher.update(str(len(raw)).encode("ascii"))
            hasher.update(b"\0")
            hasher.update(raw)
            return
        if isinstance(value, (list, tuple)):
            hasher.update(b"[\0")
            for item in value:
                _update(item)
            hasher.update(b"]\0")
            return
        raise TypeError(f"Unsupported registration payload value: {type(value)!r}")

    _update(payload)
    return hasher.hexdigest()


def _build_registration_payload(artifact: Any) -> tuple[tuple[Any, ...], str]:
    from . import _runtime_arg_registration_specs, _validate_launch_metadata

    metadata = artifact.metadata
    _validate_launch_metadata(metadata)
    runtime_entries = tuple(entry for entry in artifact.signature if not entry.is_constexpr)
    runtime_args = _runtime_arg_registration_specs(runtime_entries, metadata)

    runtime_arg_names: list[str] = []
    runtime_arg_kind_codes: list[int] = []
    runtime_arg_ranks: list[int] = []
    runtime_arg_meta_present: list[int] = []
    runtime_arg_swizzles: list[int] = []
    runtime_arg_elem_sizes: list[int] = []
    runtime_arg_elem_types: list[int] = []
    runtime_arg_fp4_padded: list[int] = []
    runtime_arg_block_size_offsets: list[int] = [0]
    runtime_arg_block_size_values: list[int] = []

    for runtime_arg in runtime_args:
        runtime_arg_names.append(str(runtime_arg["name"]))
        kind = str(runtime_arg["kind"])
        runtime_arg_kind_codes.append(_ARG_KIND_CODES[kind])
        runtime_arg_ranks.append(int(runtime_arg.get("rank", 0)))

        metadata_item = runtime_arg.get("metadata")
        if metadata_item is None:
            runtime_arg_meta_present.append(0)
            runtime_arg_swizzles.append(0)
            runtime_arg_elem_sizes.append(0)
            runtime_arg_elem_types.append(0)
            runtime_arg_fp4_padded.append(0)
        else:
            runtime_arg_meta_present.append(1)
            runtime_arg_swizzles.append(int(metadata_item["swizzle"]))
            runtime_arg_elem_sizes.append(int(metadata_item["elem_size"]))
            runtime_arg_elem_types.append(int(metadata_item["elem_type"]))
            runtime_arg_fp4_padded.append(1 if metadata_item.get("fp4_padded") else 0)
            runtime_arg_block_size_values.extend(int(value) for value in metadata_item["block_size"])
        runtime_arg_block_size_offsets.append(len(runtime_arg_block_size_values))

    payload = (
        artifact.kernel_name,
        artifact.cubin_bytes,
        32 * int(metadata["num_warps"]),
        int(metadata.get("shared", 0)),
        int(metadata.get("global_scratch_size", 0)),
        int(metadata.get("global_scratch_align", 1)),
        int(metadata.get("profile_scratch_size", 0)),
        int(metadata.get("profile_scratch_align", 1)),
        runtime_arg_names,
        runtime_arg_kind_codes,
        runtime_arg_ranks,
        runtime_arg_meta_present,
        runtime_arg_swizzles,
        runtime_arg_elem_sizes,
        runtime_arg_elem_types,
        runtime_arg_fp4_padded,
        runtime_arg_block_size_offsets,
        runtime_arg_block_size_values,
    )
    token = _registration_token_for_payload(payload)
    return payload, token


def _get_or_build_generic_launcher_module() -> tuple[str, Any]:
    from . import (
        _ensure_tvm_ffi_cache_dir,
        _maybe_load_cached_module,
        _require_tvm_ffi,
    )

    cuda_source = _get_generic_launcher_source_path().read_text()
    cache_key = hashlib.sha256(b"generic_launcher_cpp_v2\0" + cuda_source.encode("utf-8")).hexdigest()

    with _cache_lock:
        cached = _generic_module_cache.get(cache_key)
    if cached is not None:
        return cache_key, cached

    tvm_ffi_module, _ = _require_tvm_ffi()
    module_name = f"{_GENERIC_LAUNCHER_NAME}_{cache_key[:12]}"
    build_dir = _ensure_tvm_ffi_cache_dir("generic", cache_key)

    mod = _maybe_load_cached_module(tvm_ffi_module, build_dir, module_name)
    if mod is None:
        mod = _build_generic_launcher_module(tvm_ffi_module, module_name, build_dir, cuda_source)

    with _cache_lock:
        _generic_module_cache[cache_key] = mod
    return cache_key, mod


def _register_kernel_if_needed(
        launcher_cache_key: str,
        module: Any,
        registration_token: str,
        registration_args: tuple[Any, ...],
) -> int:
    cache_token = (launcher_cache_key, registration_token)
    with _cache_lock:
        cached = _registered_kernel_cache.get(cache_token)
        if cached is not None:
            return cached
        handle = int(module.register_kernel(*registration_args))
        _registered_kernel_cache[cache_token] = handle
        return handle


def _lookup_module_function(module: Any, name: str) -> Any:
    if hasattr(module, "__getitem__"):
        try:
            return module[name]
        except Exception:
            pass
    return getattr(module, name)


class TvmFfiLauncher:
    """TVM-FFI kernel launcher with the same call convention as ``CudaLauncher``."""

    def __init__(self, src, metadata, asm):
        del src

        from . import (
            _CompiledArtifact,
            _make_bound_args_launcher,
            _normalize_metadata,
            _parse_kernel_signature,
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
        self._signature = signature
        self._runtime_arg_indices = tuple(i for i, entry in enumerate(signature) if not entry.is_constexpr)
        runtime_signature = tuple(entry for entry in signature if not entry.is_constexpr)
        self._runtime_signature = runtime_signature

        artifact = _CompiledArtifact(
            kernel_name=str(metadata_dict["name"]),
            cubin_bytes=cubin_bytes,
            metadata=metadata_dict,
            signature=signature,
        )
        registration_args, registration_token = _build_registration_payload(artifact)

        launcher_cache_key, self._tvm_mod = _get_or_build_generic_launcher_module()
        self._registry_handle = _register_kernel_if_needed(
            launcher_cache_key,
            self._tvm_mod,
            registration_token,
            registration_args,
        )
        self._tvm_func = _lookup_module_function(self._tvm_mod, "launch")
        self._has_tensordesc_args = any(entry.type_name.startswith("tensordesc") for entry in runtime_signature)
        if self._has_tensordesc_args:
            self._launch_bound_args_for_tvm_ffi = _make_bound_args_launcher(
                self._tvm_func,
                self._registry_handle,
                runtime_signature,
                tvm_mod=self._tvm_mod,
            )
        else:
            self._launch_bound_args_for_tvm_ffi = None

    def launch_metadata(self, grid, stream, *args):
        return None

    def __call__(self, gridX, gridY, gridZ, stream, function, packed_metadata, launch_metadata, launch_enter_hook, launch_exit_hook, *bound_args):
        self.launch(gridX, gridY, gridZ, *bound_args)

    def _runtime_args(self, args):
        if len(args) == len(self._runtime_signature):
            return args
        if len(args) == len(self._signature):
            return tuple(args[i] for i in self._runtime_arg_indices)
        raise ValueError(
            f"Expected either {len(self._runtime_signature)} runtime args or {len(self._signature)} bound args, got {len(args)}."
        )

    def launch(self, gridX, gridY, gridZ, *args):
        args = self._runtime_args(args)
        if self._launch_bound_args_for_tvm_ffi is None:
            self._tvm_func(self._registry_handle, gridX, gridY, gridZ, *args)
            return
        self._launch_bound_args_for_tvm_ffi(gridX, gridY, gridZ, *args)
