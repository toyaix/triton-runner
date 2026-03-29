import dataclasses

import torch
import triton


FP8_DTYPE = getattr(torch, "float8_e5m2", None)
SCALAR_BLOCKS = (16, 16)
DOT_BLOCKS = (128, 64, 64)
TMA_BLOCKS = (128, 64, 64)


@dataclasses.dataclass(frozen=True)
class ArchConfig:
    variant: str
    input_dtype: torch.dtype
    output_dtype: torch.dtype
    atol: float


def arch_configs() -> dict[int, ArchConfig]:
    configs = {
        75: ArchConfig("scalar", torch.float32, torch.float32, 1e-2),
        80: ArchConfig("dot", torch.float16, torch.float32, 0.125),
        86: ArchConfig("dot", torch.float16, torch.float32, 0.125),
    }
    if FP8_DTYPE is not None:
        configs[90] = ArchConfig("tma", FP8_DTYPE, torch.float16, 0.125)
        configs[120] = ArchConfig("tma", FP8_DTYPE, torch.float16, 0.125)
    return configs


def resolve_arch_config() -> tuple[int, ArchConfig]:
    props = torch.cuda.get_device_properties(torch.device("cuda"))
    sm = props.major * 10 + props.minor
    config = arch_configs().get(sm)
    if config is None:
        supported = ", ".join(f"sm{arch}" for arch in sorted(arch_configs()))
        raise RuntimeError(f"No matmul config for sm{sm}. Supported architectures: {supported}")
    return sm, config


def enable_tma_allocator() -> None:
    def alloc_fn(size, alignment, stream):
        del alignment, stream
        return torch.empty(size, device="cuda", dtype=torch.int8)

    triton.set_allocator(alloc_fn)


def reference_matmul(a: torch.Tensor, b: torch.Tensor, config: ArchConfig) -> torch.Tensor:
    if config.variant == "tma":
        return torch.matmul(a.to(torch.float16), b.to(torch.float16))
    return torch.matmul(a, b)


def validate_matmul_outputs(ref: torch.Tensor, atol: float, *outputs: torch.Tensor) -> None:
    ref = ref.float()
    for name, out in zip(("Native Triton", "TVM-Triton", "Direct TVM-FFI"), outputs, strict=True):
        if not torch.allclose(out.float(), ref, atol=atol, rtol=0):
            raise AssertionError(f"{name} output mismatch.")
