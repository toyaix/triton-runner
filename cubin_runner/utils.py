import triton
import json
from triton.backends.nvidia.driver import make_launcher, compile_module_from_src, CudaUtils
from triton.backends.compiler import GPUTarget
from triton.compiler import make_backend
from collections import namedtuple

metadata = {}
device = triton.runtime.driver.active.get_current_device()
stream = triton.runtime.driver.active.get_current_stream(device)

def get_cufunction(json_path, cubin_path, kernel_name):
    global metadata
    metadata = json.loads(open(json_path, "r").read())
    kernel = open(cubin_path, "rb").read()
    module, function, n_regs, n_spills = triton.runtime.driver.active.utils.load_binary(
        kernel_name, kernel, metadata['shared'], device)
    return function

def get_grid_xyz(grid):
    assert grid is not None
    grid_size = len(grid)
    grid_0 = grid[0]
    grid_1 = grid[1] if grid_size > 1 else 1
    grid_2 = grid[2] if grid_size > 2 else 1
    return grid_0, grid_1, grid_2

def get_packed_metadata():
    metadata['cluster_dims'] = tuple(metadata['cluster_dims'])
    # JSON serialization dumps the target as a dict. Restore it to a GPUTarget.
    target = metadata['target']
    metadata['target'] = GPUTarget(target['backend'], target['arch'], target['warp_size'])
    KernelMetadata = namedtuple('KernelMetadata', sorted(list(metadata.keys())))
    compile_metadata = KernelMetadata(**metadata)
    backend = make_backend(compile_metadata.target)
    return backend.pack_metadata(compile_metadata)

def cubin_launch(function, signature_str, bound_args, grid):
    signature = dict(enumerate(signature_str.split()))
    src = make_launcher(None, signature)
    mod = compile_module_from_src(src, "__triton_launcher")
    global_scratch = None
    packed_metadata = get_packed_metadata()
    launch_metadata, launch_enter_hook, launch_exit_hook = None, None, None
    mod.launch(*get_grid_xyz(grid), stream, function, metadata['launch_cooperative_grid'],
               global_scratch, packed_metadata, launch_metadata, launch_enter_hook, launch_exit_hook,
               *bound_args)
