import warnings
import torch
import triton

metadata = {}

def colored_warning(message, category, filename, lineno, file=None, line=None):
    if file is None:
        import sys
        file = sys.stderr
    formatted = f"\033[1m\033[93m{category.__name__}: {message} ({filename}:{lineno})\033[0m\n"
    file.write(formatted)


warnings.showwarning = colored_warning


def check_kernel_name(kernel_name):
    if metadata['name'] != kernel_name:
        warnings.warn(
            f"This kernel name {kernel_name} is different with metadata {metadata['name']}")

def check_triton_version():
    kernel_version = metadata['triton_version']
    installed_version = triton.__version__
    if kernel_version != installed_version:
        warnings.warn(
            f"This kernel Triton v{kernel_version} is different with intstalled v{installed_version}")
    if installed_version != "3.3.1":
        warnings.warn("This runner is only support Triton v3.3.1.")


def check_cuda_arch(device):
    capability = torch.cuda.get_device_capability(device)
    capability = capability[0] * 10 + capability[1]
    kernel_arch = metadata["target"]["arch"]
    if kernel_arch != capability:
        warnings.warn(
            f"This kernel capability={kernel_arch} is different with device capability={capability}")


def check_triton(kernel_name, t_metadata, device):
    global metadata
    metadata = t_metadata
    check_kernel_name(kernel_name)
    check_triton_version()
    check_cuda_arch(device)
