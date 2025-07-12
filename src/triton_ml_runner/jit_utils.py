from triton_ml_runner.cubin_utils import get_cufunction, cubin_launch
from triton_ml_runner.compile_utils import save_cubin_from_ttir, save_cubin_from_ttgir
import os

def jit_cubin_launch(cubin_dir, kernel_name, bound_args, signature_str, grid):
    metadata_path = os.path.join(cubin_dir, f"{kernel_name}.json")
    cubin_path = os.path.join(cubin_dir, f"{kernel_name}.cubin")
    function = get_cufunction(metadata_path, cubin_path, f"{kernel_name}")
    cubin_launch(function, signature_str, bound_args, grid)


def jit_ttir_launch(file_dir, kernel_name, bound_args, signature_str, grid, options):
    ttir_path = os.path.join(file_dir, f"{kernel_name}.ttir")
    save_cubin_from_ttir(ttir_path, options, kernel_name, file_dir)
    jit_cubin_launch(file_dir, kernel_name, bound_args, signature_str, grid)


def jit_ttgir_launch(file_dir, kernel_name, bound_args, signature_str, grid, options):
    ttgir_path = os.path.join(file_dir, f"{kernel_name}.ttgir")
    save_cubin_from_ttgir(ttgir_path, options, kernel_name, file_dir)
    jit_cubin_launch(file_dir, kernel_name, bound_args, signature_str, grid)


def jit_launch(type_str, file_dir, kernel_name, bound_args, signature_str, grid, options):
    if type_str == "cubin_dir":
        jit_cubin_launch(file_dir, kernel_name, bound_args, signature_str, grid)
    elif type_str == "ttir_dir":
        jit_ttir_launch(file_dir, kernel_name, bound_args, signature_str, grid, options)
    elif type_str == "ttgir_dir":
        jit_ttgir_launch(file_dir, kernel_name, bound_args, signature_str, grid, options)
