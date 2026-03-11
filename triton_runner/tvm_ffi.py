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


def _ffi_param_decl(entry: _SignatureEntry) -> str:
    if entry.is_pointer:
        return f"tvm::ffi::TensorView {entry.name}"
    if entry.is_integer_scalar:
        return f"{_INTEGER_SCALAR_TYPES[entry.type_name]} {entry.name}"
    if entry.is_float_scalar:
        return f"double {entry.name}"
    raise NotImplementedError(f"Unsupported Triton scalar argument type: {entry.type_name}")


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


def _kernel_arg_ref(entry: _SignatureEntry) -> str:
    return f"&{entry.name}_kernel_arg"


def _render_device_binding(pointer_entries: tuple[_SignatureEntry, ...]) -> str:
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
    if pointer_entries:
        for entry in pointer_entries:
            lines.append(f'  bind_device({entry.name}.device(), "{entry.name}");')
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


def _render_launch_code(metadata: dict[str, Any], runtime_entries: tuple[_SignatureEntry, ...]) -> str:
    args_refs = ", ".join(_kernel_arg_ref(entry) for entry in runtime_entries)
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
    unsupported = [entry.type_name for entry in runtime_entries if entry.type_name.startswith("tensordesc")]
    if unsupported:
        raise NotImplementedError(
            "TVM-FFI export does not yet support tensordesc/TLX Triton signatures: "
            + ", ".join(sorted(set(unsupported))))

    embed_name = _sanitize_identifier(artifact.module_name)
    export_name = _sanitize_identifier(artifact.kernel_name)
    runtime_param_list = ", ".join(["int32_t grid_x", "int32_t grid_y", "int32_t grid_z", *[
        _ffi_param_decl(entry) for entry in runtime_entries
    ]])
    pointer_entries = tuple(entry for entry in runtime_entries if entry.is_pointer)
    arg_decls = "\n".join(f"  {_kernel_arg_decl(entry)}" for entry in runtime_entries)
    launch_code = _render_launch_code(metadata, runtime_entries)
    device_binding = _render_device_binding(pointer_entries)
    shared = int(metadata.get("shared", 0))

    return textwrap.dedent(
        f"""\
        #define TVM_FFI_CUBIN_LAUNCHER_USE_DRIVER_API 1
        #include <cuda.h>
        #include <cuda_bf16.h>
        #include <cuda_fp16.h>
        #include <cuda_runtime.h>
        #include <tvm/ffi/container/tensor.h>
        #include <tvm/ffi/error.h>
        #include <tvm/ffi/extra/c_env_api.h>
        #include <tvm/ffi/extra/cuda/cubin_launcher.h>
        #include <tvm/ffi/function.h>

        #include <cstdint>
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
