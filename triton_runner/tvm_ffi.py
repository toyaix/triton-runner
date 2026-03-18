from __future__ import annotations

import ast
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from triton.compiler.compiler import CompiledKernel

from .version_utils import is_triton_geq_v3_5, triton_version


_FLOAT_SCALAR_KERNEL_TYPES = {
    "fp16": "__half",
    "bf16": "__nv_bfloat16",
    "fp32": "float",
    "f32": "float",
    "fp64": "double",
}

_INTEGER_SCALAR_TYPES = {
    "i1": "bool",
    "i8": "int8_t",
    "i16": "int16_t",
    "i32": "int32_t",
    "i64": "int64_t",
    "u1": "bool",
    "u8": "uint8_t",
    "u16": "uint16_t",
    "u32": "uint32_t",
    "u64": "uint64_t",
}

_TENSORDESC_PATTERN = re.compile(r"tensordesc<([^[>]*)\[([^]]*)\]>")
_TMA_DTYPE_DEVICE_TO_HOST = {i: i for i in range(16)}
_TMA_DTYPE_DEVICE_TO_HOST[8] = 10
_TMA_DTYPE_DEVICE_TO_HOST[9] = 8
_TMA_DTYPE_DEVICE_TO_HOST[10] = 9
_MAX_TENSOR_DESC_RANK = 5


@dataclass(frozen=True)
class _SignatureEntry:
    name: str
    type_name: str
    specialization: Any
    is_kwargs: bool = False

    @property
    def is_constexpr(self) -> bool:
        return self.type_name == "constexpr"

    @property
    def is_pointer(self) -> bool:
        return self.type_name.startswith("*")

    @property
    def is_float_scalar(self) -> bool:
        return self.type_name in _FLOAT_SCALAR_KERNEL_TYPES

    @property
    def is_integer_scalar(self) -> bool:
        return self.type_name in _INTEGER_SCALAR_TYPES


@dataclass(frozen=True)
class _CompiledArtifact:
    kernel_name: str
    module_name: str
    cubin_bytes: bytes
    metadata: dict[str, Any]
    signature: tuple[_SignatureEntry, ...]


@dataclass(frozen=True)
class _TensorDescSpec:
    entry: _SignatureEntry
    dtype_name: str
    rank: int
    metadata: dict[str, Any] | None

    @property
    def name(self) -> str:
        return self.entry.name

    @property
    def base_name(self) -> str:
        return f"{self.name}_base"

    @property
    def padding_name(self) -> str:
        return f"{self.name}_padding_nan"

    def shape_name(self, index: int) -> str:
        return f"{self.name}_shape_{index}"

    def stride_name(self, index: int) -> str:
        return f"{self.name}_stride_{index}"


def _get_tvm_ffi_cache_dir() -> str:
    cache_dir = os.environ.get("TRITON_CACHE_DIR")
    if cache_dir is None:
        cache_dir = os.path.join(Path.home(), ".triton", "cache")
    return os.path.join(cache_dir, "tvm_ffi_launcher")


def _ensure_tvm_ffi_cache_dir(*parts: str) -> str:
    preferred = os.path.join(_get_tvm_ffi_cache_dir(), *parts)
    try:
        os.makedirs(preferred, exist_ok=True)
        return preferred
    except PermissionError:
        fallback = os.path.join(tempfile.gettempdir(), "triton_runner_cache", "tvm_ffi_launcher", *parts)
        os.makedirs(fallback, exist_ok=True)
        return fallback


def _module_name_for_metadata(metadata: dict[str, Any]) -> str:
    kernel_name = str(metadata["name"])
    hash_prefix = str(metadata.get("hash", "inline"))[:8]
    return f"triton_runner_{kernel_name}_{hash_prefix}"


def _require_tvm_ffi():
    try:
        import tvm_ffi  # type: ignore[import-not-found]
        from tvm_ffi import cpp  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(
            "triton_runner.tvm_ffi.build_module requires apache-tvm-ffi. "
            "Install it with `pip install triton-runner[tvm-ffi]` or `pip install apache-tvm-ffi`."
        ) from exc
    return tvm_ffi, cpp


def _normalize_metadata(metadata: Any) -> dict[str, Any]:
    if isinstance(metadata, dict):
        return dict(metadata)
    if hasattr(metadata, "_asdict"):
        return dict(metadata._asdict())
    fields = getattr(metadata, "_fields", None)
    if fields is not None:
        return {field: getattr(metadata, field) for field in fields}
    raise TypeError(f"Unsupported metadata object: {type(metadata)!r}")


def _parse_kernel_signature(kernel_signature: str | None) -> tuple[_SignatureEntry, ...]:
    if kernel_signature in (None, "", "None"):
        raise NotImplementedError("TVM-FFI export requires a concrete Triton kernel_signature.")
    parsed = ast.parse(kernel_signature, mode="eval").body
    if not isinstance(parsed, (ast.List, ast.Tuple)):
        raise TypeError(f"Unsupported kernel_signature shape: {type(parsed)!r}")

    entries: list[_SignatureEntry] = []
    for item in parsed.elts:
        if not isinstance(item, (ast.List, ast.Tuple)):
            raise TypeError(f"Invalid kernel_signature entry: {ast.unparse(item)!r}")
        if len(item.elts) == 4:
            name_node, type_name_node, specialization_node, is_kwargs_node = item.elts
            is_kwargs = ast.literal_eval(is_kwargs_node)
        elif len(item.elts) == 3:
            name_node, type_name_node, specialization_node = item.elts
            is_kwargs = False
        else:
            raise TypeError(
                f"Unsupported kernel_signature entry length {len(item.elts)}: {ast.unparse(item)!r}"
            )

        name = ast.literal_eval(name_node)
        type_name = ast.literal_eval(type_name_node)
        try:
            specialization = ast.literal_eval(specialization_node)
        except (TypeError, ValueError, SyntaxError):
            if type_name != "constexpr":
                raise
            # Triton may stringify constexpr specializations as Python reprs
            # such as BlockedLayout(...), which are not literal_eval-able.
            specialization = ast.unparse(specialization_node)

        entries.append(
            _SignatureEntry(
                name=str(name),
                type_name=str(type_name),
                specialization=specialization,
                is_kwargs=bool(is_kwargs),
            ))
    return tuple(entries)


def _sanitize_identifier(name: str) -> str:
    sanitized = re.sub(r"[^0-9A-Za-z_]", "_", name)
    if not sanitized:
        sanitized = "triton_runner_tvm_ffi"
    if sanitized[0].isdigit():
        sanitized = f"_{sanitized}"
    return sanitized


def _artifact_from_compiled_kernel(kernel: CompiledKernel, metadata: Any) -> _CompiledArtifact:
    asm = getattr(kernel, "asm", None)
    if asm is None or "cubin" not in asm:
        raise ValueError("TVM-FFI export requires `kernel.asm['cubin']`.")

    cubin_bytes = asm["cubin"]
    if isinstance(cubin_bytes, memoryview):
        cubin_bytes = cubin_bytes.tobytes()
    elif isinstance(cubin_bytes, bytearray):
        cubin_bytes = bytes(cubin_bytes)
    elif not isinstance(cubin_bytes, bytes):
        raise TypeError(f"Unsupported cubin type on kernel.asm['cubin']: {type(cubin_bytes)!r}")

    metadata_dict = _normalize_metadata(metadata)
    signature = _parse_kernel_signature(metadata_dict.get("kernel_signature"))
    return _CompiledArtifact(
        kernel_name=str(metadata_dict["name"]),
        module_name=_module_name_for_metadata(metadata_dict),
        cubin_bytes=cubin_bytes,
        metadata=metadata_dict,
        signature=signature,
    )


def _shared_library_path(build_dir: str | Path, module_name: str) -> Path:
    ext = ".dll" if os.name == "nt" else ".so"
    return Path(build_dir) / f"{module_name}{ext}"


def _maybe_load_cached_module(tvm_ffi_module: Any, build_dir: str | Path, module_name: str) -> Any | None:
    lib_path = _shared_library_path(build_dir, module_name)
    if not lib_path.is_file():
        return None
    return tvm_ffi_module.load_module(lib_path, keep_module_alive=True)


def _parse_tensordesc_specs(
        runtime_entries: tuple[_SignatureEntry, ...], metadata: dict[str, Any]) -> tuple[_TensorDescSpec, ...]:
    raw_metadata = metadata.get("tensordesc_meta") or ()
    specs: list[_TensorDescSpec] = []
    meta_index = 0

    for entry in runtime_entries:
        if not entry.type_name.startswith("tensordesc"):
            continue

        match = _TENSORDESC_PATTERN.fullmatch(entry.type_name)
        if match is None:
            raise TypeError(f"Unsupported tensordesc Triton signature: {entry.type_name}")

        shape_signature = match.group(2).strip()
        rank = 1 if not shape_signature else shape_signature.count(",") + 1
        if rank > _MAX_TENSOR_DESC_RANK:
            raise NotImplementedError(
                f"TVM-FFI export supports tensor descriptors up to rank {_MAX_TENSOR_DESC_RANK}, got {entry.type_name}."
            )

        spec_metadata: dict[str, Any] | None = None
        if raw_metadata:
            if meta_index >= len(raw_metadata):
                raise ValueError(
                    "Tensor descriptor metadata is shorter than the number of tensordesc runtime arguments."
                )
            raw_item = raw_metadata[meta_index]
            if raw_item is not None:
                block_size = tuple(int(v) for v in raw_item.get("block_size", ()))
                if len(block_size) != rank:
                    raise ValueError(
                        f"Tensor descriptor metadata for {entry.name} has block_size rank {len(block_size)}, expected {rank}."
                    )
                elem_type = int(raw_item["elem_type"])
                spec_metadata = {
                    "swizzle": int(raw_item["swizzle"]),
                    "elem_size": int(raw_item["elem_size"]),
                    "elem_type": int(_TMA_DTYPE_DEVICE_TO_HOST.get(elem_type, elem_type)),
                    "block_size": block_size,
                    "fp4_padded": bool(raw_item.get("fp4_padded", False)),
                }
            meta_index += 1

        specs.append(
            _TensorDescSpec(
                entry=entry,
                dtype_name=match.group(1),
                rank=rank,
                metadata=spec_metadata,
            ))

    if raw_metadata and specs and meta_index != len(raw_metadata):
        raise ValueError(
            "Tensor descriptor metadata count does not match the number of tensordesc runtime arguments."
        )

    return tuple(specs)


def _expand_tensordesc_arg_for_tvm_ffi(arg: Any, spec: _TensorDescSpec) -> list[Any]:
    missing = [field for field in ("base", "shape", "strides") if not hasattr(arg, field)]
    if missing:
        raise TypeError(
            f"Tensor descriptor argument {spec.name} must expose {', '.join(missing)} for TVM-FFI export."
        )

    shape = tuple(int(v) for v in getattr(arg, "shape"))
    strides = tuple(int(v) for v in getattr(arg, "strides"))
    if len(shape) != spec.rank:
        raise ValueError(f"Tensor descriptor argument {spec.name} expected rank {spec.rank}, got shape {shape}.")
    if len(strides) != spec.rank:
        raise ValueError(f"Tensor descriptor argument {spec.name} expected rank {spec.rank}, got strides {strides}.")

    padding = getattr(arg, "padding", None) == "nan"
    return [getattr(arg, "base"), *shape, *strides, padding]


def _expand_bound_args_for_tvm_ffi(
        signature: tuple[_SignatureEntry, ...],
        bound_args: tuple[Any, ...],
        descriptor_specs: tuple[_TensorDescSpec, ...],
) -> list[Any]:
    if len(bound_args) != len(signature):
        raise ValueError(
            f"Expected {len(signature)} bound arguments for TVM-FFI launch, got {len(bound_args)}."
        )

    expanded: list[Any] = []
    descriptor_index = 0
    for entry, arg in zip(signature, bound_args):
        if entry.is_constexpr:
            continue
        if entry.type_name.startswith("tensordesc"):
            expanded.extend(_expand_tensordesc_arg_for_tvm_ffi(arg, descriptor_specs[descriptor_index]))
            descriptor_index += 1
        else:
            expanded.append(arg)

    if descriptor_index != len(descriptor_specs):
        raise ValueError(
            "Tensor descriptor expansion count does not match the number of tensordesc runtime arguments."
        )

    return expanded


def _runtime_arg_registration_specs(
        runtime_entries: tuple[_SignatureEntry, ...],
        metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    descriptor_specs = _parse_tensordesc_specs(runtime_entries, metadata)
    descriptor_index = 0
    runtime_args: list[dict[str, Any]] = []

    for entry in runtime_entries:
        if entry.type_name.startswith("tensordesc"):
            spec = descriptor_specs[descriptor_index]
            descriptor_index += 1
            runtime_args.append({
                "name": spec.name,
                "kind": "tensordesc",
                "rank": spec.rank,
                "metadata": spec.metadata,
            })
            continue

        if entry.is_pointer:
            kind = "pointer"
        elif entry.is_integer_scalar or entry.is_float_scalar:
            kind = entry.type_name
        else:
            raise NotImplementedError(f"Unsupported Triton runtime argument type: {entry.type_name}")

        runtime_args.append({"name": entry.name, "kind": kind})

    return runtime_args


def _validate_launch_metadata(metadata: dict[str, Any]) -> None:
    cluster_dims = tuple(int(v) for v in metadata.get("cluster_dims", (1, 1, 1)))
    cluster_product = cluster_dims[0] * cluster_dims[1] * cluster_dims[2]
    unsupported: list[str] = []
    if cluster_product != 1 or int(metadata.get("num_ctas", 1)) != 1:
        unsupported.append("cluster launch / num_ctas != 1")
    if metadata.get("launch_pdl"):
        unsupported.append("launch_pdl")
    if metadata.get("launch_cooperative_grid"):
        unsupported.append("launch_cooperative_grid")
    if unsupported:
        raise NotImplementedError(
            "Current apache-tvm-ffi releases expose only `CubinKernel.Launch(...)`; "
            "the Triton metadata uses unsupported launch features: " + ", ".join(unsupported)
        )
