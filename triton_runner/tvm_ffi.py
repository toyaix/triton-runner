from __future__ import annotations

import ast
import os
import re
import shutil
import textwrap
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


def _ffi_param_decl(entry: _SignatureEntry) -> str:
    if entry.is_pointer:
        return f"tvm::ffi::TensorView {entry.name}"
    if entry.is_integer_scalar:
        return f"{_INTEGER_SCALAR_TYPES[entry.type_name]} {entry.name}"
    if entry.is_float_scalar:
        return f"double {entry.name}"
    raise NotImplementedError(f"Unsupported Triton scalar argument type: {entry.type_name}")


def _tensordesc_ffi_param_decls(spec: _TensorDescSpec) -> list[str]:
    return [
        f"tvm::ffi::TensorView {spec.base_name}",
        *[f"int64_t {spec.shape_name(index)}" for index in range(spec.rank)],
        *[f"int64_t {spec.stride_name(index)}" for index in range(spec.rank)],
        f"bool {spec.padding_name}",
    ]


def _kernel_arg_decl(entry: _SignatureEntry) -> str:
    local_name = f"{entry.name}_kernel_arg"
    if entry.is_pointer:
        return f"void* {local_name} = {entry.name}.data_ptr();"
    if entry.is_integer_scalar:
        return f"{_INTEGER_SCALAR_TYPES[entry.type_name]} {local_name} = {entry.name};"
    if entry.type_name == "fp16":
        return f"__half {local_name} = __float2half(static_cast<float>({entry.name}));"
    if entry.type_name == "bf16":
        return f"__nv_bfloat16 {local_name} = __float2bfloat16(static_cast<float>({entry.name}));"
    if entry.type_name in {"fp32", "f32"}:
        return f"float {local_name} = static_cast<float>({entry.name});"
    if entry.type_name == "fp64":
        return f"double {local_name} = {entry.name};"
    raise NotImplementedError(f"Unsupported Triton scalar argument type: {entry.type_name}")


def _render_tensordesc_setup(spec: _TensorDescSpec) -> tuple[list[str], list[str]]:
    lines = [f"void* {spec.name}_base_kernel_arg = {spec.base_name}.data_ptr();"]
    shape_i64_names: list[str] = []
    shape_i32_names: list[str] = []
    stride_names: list[str] = []
    arg_refs: list[str] = []

    for index in range(spec.rank):
        shape_name = spec.shape_name(index)
        stride_name = spec.stride_name(index)
        shape_i64_name = f"{shape_name}_i64_kernel_arg"
        shape_i32_name = f"{shape_name}_i32_kernel_arg"
        stride_kernel_name = f"{stride_name}_kernel_arg"

        lines.extend([
            f"TVM_FFI_CHECK({shape_name} >= 0 && {shape_name} <= static_cast<int64_t>(std::numeric_limits<int32_t>::max()), ValueError)",
            f'    << "Tensor descriptor {spec.name} shape[{index}] must fit in int32.";'
        ])
        lines.append(f"int64_t {shape_i64_name} = {shape_name};")
        lines.append(f"int32_t {shape_i32_name} = static_cast<int32_t>({shape_name});")
        lines.append(f"int64_t {stride_kernel_name} = {stride_name};")

        shape_i64_names.append(shape_i64_name)
        shape_i32_names.append(shape_i32_name)
        stride_names.append(stride_kernel_name)

    padding_kernel_name = f"{spec.name}_padding_kernel_arg"
    lines.append(f"bool {padding_kernel_name} = {spec.padding_name};")

    if spec.metadata is not None:
        block_size_literal = ", ".join(f"static_cast<uint32_t>({value})" for value in spec.metadata["block_size"])
        lines.append(f"int64_t {spec.name}_shape_values[{spec.rank}] = {{{', '.join(shape_i64_names)}}};")
        lines.append(f"int64_t {spec.name}_stride_values[{spec.rank}] = {{{', '.join(stride_names)}}};")
        lines.append(f"uint32_t {spec.name}_block_size[{spec.rank}] = {{{block_size_literal}}};")
        lines.append(f"CUtensorMap {spec.name}_tensor_map{{}};")
        lines.append(
            f"FillTmaDescriptor(&{spec.name}_tensor_map, {spec.base_name}.data_ptr(), "
            f"{spec.metadata['swizzle']}, {spec.metadata['elem_size']}, {spec.metadata['elem_type']}, "
            f"{spec.name}_block_size, {spec.rank}, {spec.name}_shape_values, {spec.name}_stride_values, "
            f"{padding_kernel_name}, {'true' if spec.metadata['fp4_padded'] else 'false'}, \"{spec.name}\");"
        )
        arg_refs.append(f"&{spec.name}_tensor_map")
        arg_refs.extend(f"&{shape_i32_name}" for shape_i32_name in shape_i32_names)
        arg_refs.extend(f"&{stride_kernel_name}" for stride_kernel_name in stride_names)
    else:
        arg_refs.append(f"&{spec.name}_base_kernel_arg")
        arg_refs.extend(f"&{shape_i64_name}" for shape_i64_name in shape_i64_names)
        arg_refs.extend(f"&{stride_kernel_name}" for stride_kernel_name in stride_names)
        arg_refs.append(f"&{padding_kernel_name}")
        arg_refs.extend(f"&{shape_i32_name}" for shape_i32_name in shape_i32_names)
        arg_refs.extend(f"&{stride_kernel_name}" for stride_kernel_name in stride_names)

    return lines, arg_refs


def _render_runtime_args(
        runtime_entries: tuple[_SignatureEntry, ...],
        metadata: dict[str, Any],
) -> tuple[tuple[_TensorDescSpec, ...], list[str], list[tuple[str, str]], list[str], list[str]]:
    descriptor_specs = _parse_tensordesc_specs(runtime_entries, metadata)
    descriptor_index = 0
    runtime_param_decls: list[str] = []
    tensor_bindings: list[tuple[str, str]] = []
    arg_decls: list[str] = []
    arg_refs: list[str] = []

    for entry in runtime_entries:
        if entry.type_name.startswith("tensordesc"):
            spec = descriptor_specs[descriptor_index]
            descriptor_index += 1
            runtime_param_decls.extend(_tensordesc_ffi_param_decls(spec))
            tensor_bindings.append((spec.base_name, spec.name))
            tensordesc_decls, tensordesc_refs = _render_tensordesc_setup(spec)
            arg_decls.extend(tensordesc_decls)
            arg_refs.extend(tensordesc_refs)
            continue

        runtime_param_decls.append(_ffi_param_decl(entry))
        if entry.is_pointer:
            tensor_bindings.append((entry.name, entry.name))
        arg_decls.append(_kernel_arg_decl(entry))
        arg_refs.append(f"&{entry.name}_kernel_arg")

    return descriptor_specs, runtime_param_decls, tensor_bindings, arg_decls, arg_refs


def _render_device_binding(tensor_bindings: tuple[tuple[str, str], ...]) -> str:
    lines = [
        "  DLDevice device{};",
        "  bool device_initialized = false;",
        "  auto bind_device = [&](DLDevice candidate, const char* arg_name) {",
        "    TVM_FFI_CHECK(candidate.device_type == kDLCUDA, ValueError)",
        '        << "TVM-FFI Triton export only supports CUDA tensors, got device_type="',
        '        << candidate.device_type << " for argument " << arg_name;',
        "    if (!device_initialized) {",
        "      device = candidate;",
        "      device_initialized = true;",
        "      return;",
        "    }",
        "    TVM_FFI_CHECK(device.device_type == candidate.device_type && device.device_id == candidate.device_id, ValueError)",
        '        << "All tensor arguments must live on the same CUDA device.";',
        "  };",
    ]
    if tensor_bindings:
        for param_name, display_name in tensor_bindings:
            lines.append(f'  bind_device({param_name}.device(), "{display_name}");')
    else:
        lines.extend([
            "  int current_device = 0;",
            "  TVM_FFI_CHECK_TRITON_RUNNER_CUDA_RUNTIME_ERROR(cudaGetDevice(&current_device));",
            "  device.device_type = kDLCUDA;",
            "  device.device_id = current_device;",
            "  device_initialized = true;",
        ])
    return "\n".join(lines)


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


def _render_attr_setup(metadata: dict[str, Any]) -> str:
    shared = int(metadata.get("shared", 0))

    lines = [
        "  tvm::ffi::dim3 grid(static_cast<unsigned>(grid_x), static_cast<unsigned>(grid_y), static_cast<unsigned>(grid_z));",
        f"  tvm::ffi::dim3 block(static_cast<unsigned>(32 * {int(metadata['num_warps'])}), 1u, 1u);",
        "  cudaStream_t stream = static_cast<cudaStream_t>(TVMFFIEnvGetStream(device.device_type, device.device_id));",
        f"  constexpr uint32_t shared_memory = static_cast<uint32_t>({shared});",
        "  int previous_device = -1;",
        "  TVM_FFI_CHECK_TRITON_RUNNER_CUDA_RUNTIME_ERROR(cudaGetDevice(&previous_device));",
        "  if (previous_device != device.device_id) {",
        "    TVM_FFI_CHECK_TRITON_RUNNER_CUDA_RUNTIME_ERROR(cudaSetDevice(device.device_id));",
        "  }",
    ]
    return "\n".join(lines)


def _render_tensordesc_helpers(has_tensordesc: bool) -> str:
    if not has_tensordesc:
        return ""

    return textwrap.dedent(
        """\
        inline void CheckCudaDriverError(CUresult err) {
          if (err != CUDA_SUCCESS) {
            const char* err_name = nullptr;
            const char* err_string = nullptr;
            cuGetErrorName(err, &err_name);
            cuGetErrorString(err, &err_string);
            TVM_FFI_THROW(RuntimeError)
                << "CUDA Driver Error: " << (err_name != nullptr ? err_name : "<unknown>")
                << " (" << static_cast<int>(err) << "): "
                << (err_string != nullptr ? err_string : "<unknown>");
          }
        }

        #define TVM_FFI_CHECK_TRITON_RUNNER_CUDA_DRIVER_ERROR(stmt) \\
          do { \\
            ::CUresult __err = (stmt); \\
            ::triton_runner_tvm_ffi::CheckCudaDriverError(__err); \\
          } while (0)

        using cuTensorMapEncodeTiled_t = CUresult (*)(
            CUtensorMap*,
            CUtensorMapDataType,
            cuuint32_t,
            void*,
            const cuuint64_t*,
            const cuuint64_t*,
            const cuuint32_t*,
            const cuuint32_t*,
            CUtensorMapInterleave,
            CUtensorMapSwizzle,
            CUtensorMapL2promotion,
            CUtensorMapFloatOOBfill);

        inline cuTensorMapEncodeTiled_t GetCuTensorMapEncodeTiledHandle() {
          static void* lib_handle = nullptr;
          static cuTensorMapEncodeTiled_t func = nullptr;
          if (func != nullptr) {
            return func;
          }
          if (lib_handle == nullptr) {
            lib_handle = dlopen("libcuda.so.1", RTLD_LAZY | RTLD_LOCAL);
            TVM_FFI_CHECK(lib_handle != nullptr, RuntimeError)
                << "Failed to open libcuda.so.1 for tensor descriptor support.";
          }
          dlerror();
          func = reinterpret_cast<cuTensorMapEncodeTiled_t>(dlsym(lib_handle, "cuTensorMapEncodeTiled"));
          const char* err = dlerror();
          TVM_FFI_CHECK(err == nullptr && func != nullptr, RuntimeError)
              << "Failed to retrieve cuTensorMapEncodeTiled from libcuda.so.1.";
          return func;
        }

        inline void FillTmaDescriptor(CUtensorMap* tensor_map,
                                      void* global_address,
                                      int swizzle,
                                      int elem_size,
                                      int elem_type,
                                      const uint32_t* block_size,
                                      int rank,
                                      const int64_t* shape,
                                      const int64_t* strides,
                                      bool padding_nan,
                                      bool fp4_padded,
                                      const char* arg_name) {
          TVM_FFI_CHECK(rank > 0 && rank <= 5, ValueError)
              << "Tensor descriptor " << arg_name << " has unsupported rank " << rank;
          TVM_FFI_CHECK(strides[rank - 1] == 1, ValueError)
              << "Tensor descriptor " << arg_name << " requires innermost stride == 1.";

          uint32_t block_size_int[5] = {1, 1, 1, 1, 1};
          uint64_t shape_int[5] = {1, 1, 1, 1, 1};
          uint64_t strides_bytes[5] = {0, 0, 0, 0, 0};
          uint32_t element_strides[5] = {1, 1, 1, 1, 1};

          for (int i = 0; i < rank; ++i) {
            TVM_FFI_CHECK(shape[i] >= 0, ValueError)
                << "Tensor descriptor " << arg_name << " shape[" << i << "] must be non-negative.";
            int reversed = rank - i - 1;
            uint64_t dim = static_cast<uint64_t>(shape[i]);
            if (fp4_padded && i == rank - 1) {
              dim *= 2;
            }
            shape_int[reversed] = dim;
            block_size_int[reversed] = block_size[i];
          }

          for (int i = 0; i + 1 < rank; ++i) {
            TVM_FFI_CHECK(strides[i] >= 0, ValueError)
                << "Tensor descriptor " << arg_name << " stride[" << i << "] must be non-negative.";
            int reversed = rank - i - 2;
            strides_bytes[reversed] = static_cast<uint64_t>(elem_size) * static_cast<uint64_t>(strides[i]);
          }
          strides_bytes[rank - 1] =
              shape_int[rank - 1] * static_cast<uint64_t>(rank == 1 ? elem_size : strides_bytes[rank - 2]);

          CUtensorMapFloatOOBfill fill =
              padding_nan ? CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA : CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
          auto encode = GetCuTensorMapEncodeTiledHandle();
          auto result = encode(
              tensor_map,
              static_cast<CUtensorMapDataType>(elem_type),
              static_cast<cuuint32_t>(rank),
              global_address,
              shape_int,
              strides_bytes,
              block_size_int,
              element_strides,
              CU_TENSOR_MAP_INTERLEAVE_NONE,
              static_cast<CUtensorMapSwizzle>(swizzle),
              CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
              fill);
          TVM_FFI_CHECK_TRITON_RUNNER_CUDA_DRIVER_ERROR(result);
        }
        """
    )


def _render_launch_code(metadata: dict[str, Any], arg_refs: tuple[str, ...]) -> str:
    args_refs = ", ".join(arg_refs)
    args_refs = f"{args_refs}, " if args_refs else ""
    cluster_dims = tuple(int(v) for v in metadata.get("cluster_dims", (1, 1, 1)))
    cluster_cta_count = cluster_dims[0] * cluster_dims[1] * cluster_dims[2]

    return textwrap.dedent(
        f"""\
          void* global_scratch = GetScratchBuffer(
              g_global_scratch_buffers,
              device.device_id,
              static_cast<size_t>(grid_x) * static_cast<size_t>(grid_y) * static_cast<size_t>(grid_z) *
                  static_cast<size_t>({cluster_cta_count}) * static_cast<size_t>({int(metadata.get("global_scratch_size", 0))}),
              static_cast<size_t>({int(metadata.get("global_scratch_align", 1))}));
          void* profile_scratch = GetScratchBuffer(
              g_profile_scratch_buffers,
              device.device_id,
              static_cast<size_t>(grid_x) * static_cast<size_t>(grid_y) * static_cast<size_t>(grid_z) *
                  static_cast<size_t>({cluster_cta_count}) * static_cast<size_t>({int(metadata.get("profile_scratch_size", 0))}),
              static_cast<size_t>({int(metadata.get("profile_scratch_align", 1))}));

          void* args[] = {{{args_refs}&global_scratch, &profile_scratch}};
          if (grid_x > 0 && grid_y > 0 && grid_z > 0) {{
            auto result = kernel.Launch(args, grid, block, stream, shared_memory);
            TVM_FFI_CHECK_CUBIN_LAUNCHER_CUDA_ERROR(result);
          }}

          if (previous_device != device.device_id) {{
            TVM_FFI_CHECK_TRITON_RUNNER_CUDA_RUNTIME_ERROR(cudaSetDevice(previous_device));
          }}
        """
    )


def _render_cuda_shim(artifact: _CompiledArtifact) -> str:
    metadata = artifact.metadata
    runtime_entries = tuple(entry for entry in artifact.signature if not entry.is_constexpr)
    _validate_launch_metadata(metadata)
    descriptor_specs, runtime_param_decls, tensor_bindings, arg_setup_lines, arg_refs = _render_runtime_args(
        runtime_entries, metadata)

    embed_name = _sanitize_identifier(artifact.module_name)
    export_name = _sanitize_identifier(artifact.kernel_name)
    runtime_param_list = ", ".join(["int32_t grid_x", "int32_t grid_y", "int32_t grid_z", *runtime_param_decls])
    arg_decls = "\n".join(f"  {line}" for line in arg_setup_lines)
    launch_code = _render_launch_code(metadata, tuple(arg_refs))
    device_binding = _render_device_binding(tuple(tensor_bindings))
    shared = int(metadata.get("shared", 0))
    tensordesc_helpers = _render_tensordesc_helpers(bool(descriptor_specs))

    return textwrap.dedent(
        f"""\
        #define TVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API 1
        #include <cuda.h>
        #include <cuda_bf16.h>
        #include <cuda_fp16.h>
        #include <cuda_runtime.h>
        #include <dlfcn.h>
        #include <tvm/ffi/container/tensor.h>
        #include <tvm/ffi/error.h>
        #include <tvm/ffi/extra/c_env_api.h>
        #include <tvm/ffi/extra/cuda/cubin_launcher.h>
        #include <tvm/ffi/function.h>

        #include <cstdint>
        #include <limits>
        #include <mutex>
        #include <unordered_map>

        TVM_FFI_EMBED_CUBIN({embed_name});

        namespace triton_runner_tvm_ffi {{

        inline void CheckCudaRuntimeError(cudaError_t err) {{
          if (err != cudaSuccess) {{
            TVM_FFI_THROW(RuntimeError)
                << "CUDA Runtime Error: " << cudaGetErrorName(err) << " ("
                << static_cast<int>(err) << "): " << cudaGetErrorString(err);
          }}
        }}

        #define TVM_FFI_CHECK_TRITON_RUNNER_CUDA_RUNTIME_ERROR(stmt) \\
          do {{ \\
            ::cudaError_t __err = (stmt); \\
            ::triton_runner_tvm_ffi::CheckCudaRuntimeError(__err); \\
          }} while (0)

        {tensordesc_helpers.rstrip()}

        struct ScratchBuffer {{
          void* base = nullptr;
          void* aligned = nullptr;
          size_t capacity = 0;
          size_t alignment = 1;
        }};

        static std::mutex g_scratch_mu;
        static std::unordered_map<int, ScratchBuffer> g_global_scratch_buffers;
        static std::unordered_map<int, ScratchBuffer> g_profile_scratch_buffers;

        inline void* AlignPtr(void* ptr, size_t alignment) {{
          uintptr_t raw = reinterpret_cast<uintptr_t>(ptr);
          uintptr_t mask = alignment - 1;
          uintptr_t aligned = (raw + mask) & ~mask;
          return reinterpret_cast<void*>(aligned);
        }}

        inline void* GetScratchBuffer(std::unordered_map<int, ScratchBuffer>& buffers,
                                      int device_id,
                                      size_t size,
                                      size_t alignment) {{
          if (size == 0) {{
            return nullptr;
          }}
          if (alignment == 0) {{
            alignment = 1;
          }}
          std::lock_guard<std::mutex> guard(g_scratch_mu);
          auto& buffer = buffers[device_id];
          if (buffer.base == nullptr || buffer.capacity < size || buffer.alignment < alignment) {{
            int previous_device = -1;
            TVM_FFI_CHECK_TRITON_RUNNER_CUDA_RUNTIME_ERROR(cudaGetDevice(&previous_device));
            if (previous_device != device_id) {{
              TVM_FFI_CHECK_TRITON_RUNNER_CUDA_RUNTIME_ERROR(cudaSetDevice(device_id));
            }}
            if (buffer.base != nullptr) {{
              TVM_FFI_CHECK_TRITON_RUNNER_CUDA_RUNTIME_ERROR(cudaFree(buffer.base));
            }}
            void* base = nullptr;
            TVM_FFI_CHECK_TRITON_RUNNER_CUDA_RUNTIME_ERROR(cudaMalloc(&base, size + alignment - 1));
            buffer.base = base;
            buffer.aligned = AlignPtr(base, alignment);
            buffer.capacity = size;
            buffer.alignment = alignment;
            if (previous_device != device_id) {{
              TVM_FFI_CHECK_TRITON_RUNNER_CUDA_RUNTIME_ERROR(cudaSetDevice(previous_device));
            }}
          }}
          return buffer.aligned;
        }}

        void {export_name}({runtime_param_list}) {{
          static auto kernel =
              EmbedCubinModule_{embed_name}::Global()->mod.GetKernelWithMaxDynamicSharedMemory(
                  "{artifact.kernel_name}", {shared});
        {device_binding}
        {_render_attr_setup(metadata)}
        {arg_decls}
        {launch_code.rstrip()}
        }}

        }}  // namespace triton_runner_tvm_ffi

        TVM_FFI_DLL_EXPORT_TYPED_FUNC({export_name}, triton_runner_tvm_ffi::{export_name});
        """
    )
