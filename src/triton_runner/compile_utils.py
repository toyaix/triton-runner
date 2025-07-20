import triton
import hashlib
import os
import re
from .check_utils import check_cuda_arch_with_capability
from .make_ptx_utils import runner_make_ptx
from .make_cubin_utils import runner_make_cubin

_context = triton._C.libtriton.ir.context()
_target = triton.runtime.driver.active.get_current_target()
_backend = triton.compiler.make_backend(_target)
_src = None


def update_src(full_path):
    global _src
    _src = triton.compiler.IRSource(full_path, _context, _backend)

def get_satges(options):
    stages = {}
    _backend.add_stages(stages, options)
    return stages


def load_context():
    triton._C.libtriton.ir.load_dialects(_context)
    _backend.load_dialects(_context)


def get_metadata(options):
    env_vars = triton._C.libtriton.get_cache_invalidating_env_vars()
    key = f"{_src.hash()}-{_backend.hash()}-{options.hash()}-{str(sorted(env_vars.items()))}"
    hash = "runner-" + hashlib.sha256(key.encode("utf-8")).hexdigest()
    metadata = {
        "hash": hash,
        "target": _target,
        **options.__dict__,
        **env_vars,
    }
    metadata["triton_version"] = triton.__version__
    return metadata


def get_module(options):
    # module = triton._C.libtriton.ir.parse_mlir_module(full_path, context)
    codegen_fns = _backend.get_codegen_implementation(options)
    module_map = _backend.get_module_map()
    module = _src.make_ir(options, codegen_fns, module_map, _context)
    return module


def save_cubin_from_ir_with_first_stage(full_path, options, kernel_name, save_path, first_stage):
    update_src(full_path)
    stages = get_satges(options)
    metadata = get_metadata(options)
    module = get_module(options)
    for ext, compile_ir in list(stages.items())[first_stage:]:
        next_module = compile_ir(module, metadata)
        module = next_module
    cubin_path = os.path.join(save_path, f"{kernel_name}.cubin")
    metadata_path = os.path.join(save_path, f"{kernel_name}.json")
    with open(cubin_path, "wb") as cubin_file, open(metadata_path, "w") as metadata_file:
        cubin_file.write(module)
        import json
        metadata_file.write(json.dumps(metadata, default=vars))


def save_cubin_from_ttir(full_path, t_options, kernel_name, save_path):
    save_cubin_from_ir_with_first_stage(full_path, t_options, kernel_name, save_path, 1)


def save_cubin_from_ttgir(full_path, t_options, kernel_name, save_path):
    save_cubin_from_ir_with_first_stage(full_path, t_options, kernel_name, save_path, 2)


def save_cubin_from_ptx(full_path, kernel_name, save_path):
    target = triton.runtime.driver.active.get_current_target()
    backend = triton.compiler.make_backend(target)
    options = backend.parse_options(dict())
    src = open(full_path, 'r').read()
    target_match = re.search(r"\.target\s+sm_(\d+)", src)
    ptx_capability = target_match.group(1)
    check_cuda_arch_with_capability(int(ptx_capability), target.arch)
    cubin = runner_make_cubin(src, options, target.arch)
    cubin_path = os.path.join(save_path, f"{kernel_name}.cubin")
    with open(cubin_path, "wb") as cubin_file:
        cubin_file.write(cubin)


def save_cubin_from_llir(full_path, kernel_name, save_path):
    target = triton.runtime.driver.active.get_current_target()
    backend = triton.compiler.make_backend(target)
    options = backend.parse_options(dict())
    src = open(full_path, 'r').read()
    ptx = runner_make_ptx(src, options, target.arch)
    ptx_path = os.path.join(save_path, f"{kernel_name}.ptx")
    with open(ptx_path, "w") as ptx_file:
        ptx_file.write(ptx)
    save_cubin_from_ptx(ptx_path, kernel_name, save_path)
