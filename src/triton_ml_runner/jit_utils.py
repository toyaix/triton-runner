from triton_ml_runner.cubin_utils import get_cufunction, cubin_launch
import os

def jit_cubin_launch(cubin_dir, kernel_name, bound_args, signature_str, grid):
    metadata_path = os.path.join(cubin_dir, f"{kernel_name}.json")
    cubin_path = os.path.join(cubin_dir, f"{kernel_name}.cubin")
    function = get_cufunction(metadata_path, cubin_path, f"{kernel_name}")
    cubin_launch(function, signature_str, bound_args, grid)

def jit_launch(type_str, file_dir, kernel_name, bound_args, signature_str, grid):
    if type_str == "cubin_dir":
        jit_cubin_launch(file_dir, kernel_name, bound_args, signature_str, grid)