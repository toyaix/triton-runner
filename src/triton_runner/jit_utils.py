from .cubin_utils import get_cufunction, cubin_launch_config, kernel_launch_config
from .compile_utils import *
from .components import KernelLauncher
import os

_kernel_launcher = None


def jit_cubin_launch(cubin_dir, kernel_name, bound_args, signature_str, grid):
    metadata_path = os.path.join(cubin_dir, f"{kernel_name}.json")
    cubin_path = os.path.join(cubin_dir, f"{kernel_name}.cubin")
    function = get_cufunction(metadata_path, cubin_path, f"{kernel_name}")
    global _kernel_launcher
    _kernel_launcher = KernelLauncher(*cubin_launch_config(function, signature_str, bound_args, grid))


def jit_kerel_launch(kernel, sigvals, bound_args, grid):
    signature_str = " ".join(sigvals)
    return KernelLauncher(*kernel_launch_config(kernel, signature_str, bound_args, grid))


def jit_ttir_launch(file_dir, kernel_name, bound_args, signature_str, grid, options):
    ttir_path = os.path.join(file_dir, f"{kernel_name}.ttir")
    save_cubin_from_ttir(ttir_path, options, kernel_name, file_dir)
    jit_cubin_launch(file_dir, kernel_name, bound_args, signature_str, grid)


def jit_ttgir_launch(file_dir, kernel_name, bound_args, signature_str, grid, options):
    ttgir_path = os.path.join(file_dir, f"{kernel_name}.ttgir")
    save_cubin_from_ttgir(ttgir_path, options, kernel_name, file_dir)
    jit_cubin_launch(file_dir, kernel_name, bound_args, signature_str, grid)


def jit_llir_launch(file_dir, kernel_name, bound_args, signature_str, grid):
    llir_path = os.path.join(file_dir, f"{kernel_name}.llir")
    save_cubin_from_llir(llir_path, kernel_name, file_dir)
    jit_cubin_launch(file_dir, kernel_name, bound_args, signature_str, grid)


def jit_ptx_launch(file_dir, kernel_name, bound_args, signature_str, grid):
    ptx_path = os.path.join(file_dir, f"{kernel_name}.ptx")
    save_cubin_from_ptx(ptx_path, kernel_name, file_dir)
    jit_cubin_launch(file_dir, kernel_name, bound_args, signature_str, grid)


def jit_launch(type_str, file_dir, kernel_name, bound_args, signature_str, grid, options):
    if type_str == "cubin":
        jit_cubin_launch(file_dir, kernel_name, bound_args, signature_str, grid)
    elif type_str == "ttir":
        jit_ttir_launch(file_dir, kernel_name, bound_args, signature_str, grid, options)
    elif type_str == "ttgir":
        jit_ttgir_launch(file_dir, kernel_name, bound_args, signature_str, grid, options)
    elif type_str == "llir":
        jit_llir_launch(file_dir, kernel_name, bound_args, signature_str, grid)
    elif type_str == "ptx":
        jit_ptx_launch(file_dir, kernel_name, bound_args, signature_str, grid)
    return _kernel_launcher
