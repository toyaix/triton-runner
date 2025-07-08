import triton
import json
from triton.backends.nvidia.driver import make_launcher, compile_module_from_src
from collections import namedtuple
import warnings
import torch

metadata = {}
device = triton.runtime.driver.active.get_current_device()
stream = triton.runtime.driver.active.get_current_stream(device)


def colored_warning(message, category, filename, lineno, file=None, line=None):
    if file is None:
        import sys
        file = sys.stderr
    formatted = f"\033[1m\033[93m{category.__name__}: {message} ({filename}:{lineno})\033[0m\n"
    file.write(formatted)


warnings.showwarning = colored_warning


def check_triton_version():
    if metadata['triton_version'] != "3.3.1":
        warnings.warn("This runner is only support Triton v3.3.1.")


def check_cuda_arch():
    capability = torch.cuda.get_device_capability(device)
    capability = capability[0] * 10 + capability[1]
    kernel_arch = metadata["target"]["arch"]
    if kernel_arch != capability:
        warnings.warn(
            f"This kernel capability={kernel_arch} is different with device={capability}")


def check_triton():
    check_triton_version()
    check_cuda_arch()


def get_cufunction(json_path, cubin_path, kernel_name):
    global metadata
    metadata = json.loads(open(json_path, "r").read())
    check_triton()
    kernel = open(cubin_path, "rb").read()
    module, function, n_regs, n_spills = triton.runtime.driver.active.utils.load_binary(
        kernel_name, kernel, metadata["shared"], device)
    return function


def get_grid_xyz(grid):
    assert grid is not None
    grid_size = len(grid)
    grid_0 = grid[0]
    grid_1 = grid[1] if grid_size > 1 else 1
    grid_2 = grid[2] if grid_size > 2 else 1
    return grid_0, grid_1, grid_2


def get_packed_metadata():
    metadata["cluster_dims"] = tuple(metadata["cluster_dims"])
    # JSON serialization dumps the target as a dict. Restore it to a GPUTarget.
    target = metadata["target"]
    metadata["target"] = triton.backends.compiler.GPUTarget(
        target["backend"], target["arch"], target["warp_size"])
    KernelMetadata = namedtuple(
        "KernelMetadata", sorted(list(metadata.keys())))
    compile_metadata = KernelMetadata(**metadata)
    backend = triton.compiler.make_backend(compile_metadata.target)
    return backend.pack_metadata(compile_metadata)


def get_global_scratch(grid):
    if metadata["global_scratch_size"] > 0:
        gridX, gridY, gridZ = get_grid_xyz(grid)
        grid_size = gridX * gridY * gridZ
        alloc_size = grid_size * metadata["global_scratch_size"]
        return triton.runtime._allocation._allocator(alloc_size, metadata["global_scratch_align"], stream)
    return None


def cubin_launch(function, signature_str, bound_args, grid):
    signature = dict(enumerate(signature_str.split()))
    src = make_launcher(None, signature)
    mod = compile_module_from_src(src, "__triton_launcher")
    global_scratch = get_global_scratch(grid)
    packed_metadata = get_packed_metadata()
    launch_metadata, launch_enter_hook, launch_exit_hook = None, None, None
    mod.launch(*get_grid_xyz(grid), stream, function,
               metadata["launch_cooperative_grid"], global_scratch,
               packed_metadata, launch_metadata, launch_enter_hook, launch_exit_hook, *bound_args)
