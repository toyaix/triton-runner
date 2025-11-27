import warnings
import triton
from .color_print import get_project_name

_metadata = {}


def colored_warning(message, category, filename, lineno, file=None, line=None):
    if file is None:
        import sys
        file = sys.stderr
    formatted = f"\033[1m\033[93m{category.__name__}: {message} ({filename}:{lineno})\033[0m\n"
    file.write(formatted)


warnings.showwarning = colored_warning


def check_kernel_name(kernel_name):
    if _metadata['name'] != kernel_name:
        warnings.warn(f"{get_project_name()} This kernel name {kernel_name} is different with metadata {_metadata['name']}")


def check_triton_version():
    kernel_version = _metadata.get('triton_version', '')
    from .version_utils import triton_version
    installed_version = triton_version
    if kernel_version and kernel_version != installed_version:
        warnings.warn(f"{get_project_name()} This kernel Triton v{kernel_version} is different with intstalled v{installed_version}")


def check_cuda_arch_with_capability(kernel_arch, target_arch):
    if kernel_arch != target_arch:
        warnings.warn(f"{get_project_name()} This kernel capability={kernel_arch} is different with device capability={target_arch}")


def check_cuda_arch(target):
    kernel_arch = _metadata["target"]["arch"]
    check_cuda_arch_with_capability(target.arch, kernel_arch)


def runner_check_triton(kernel_name, metadata, target):
    global _metadata
    _metadata = metadata
    check_kernel_name(kernel_name)
    check_triton_version()
    check_cuda_arch(target)
