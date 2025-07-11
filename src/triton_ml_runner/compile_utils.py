import triton
import hashlib
import os

context = triton._C.libtriton.ir.context()
target = triton.runtime.driver.active.get_current_target()
backend = triton.compiler.make_backend(target)
src = None


def get_options(full_path):
    global src
    src = triton.compiler.IRSource(full_path, context, backend)
    extra_options = src.parse_options()
    options = backend.parse_options(dict(None or dict(), **extra_options))
    return options


def get_satges(options):
    stages = {}
    backend.add_stages(stages, options)
    return stages


def load_context():
    triton._C.libtriton.ir.load_dialects(context)
    backend.load_dialects(context)


def get_metadata(options):
    env_vars = triton._C.libtriton.get_cache_invalidating_env_vars()
    key = f"{src.hash()}-{backend.hash()}-{options.hash()}-{str(sorted(env_vars.items()))}"
    hash = "mlir_runner-" + hashlib.sha256(key.encode("utf-8")).hexdigest()
    metadata = {
        "hash": hash,
        "target": target,
        **options.__dict__,
        **env_vars,
    }
    metadata["triton_version"] = triton.__version__
    return metadata


def get_module(options):
    # module = triton._C.libtriton.ir.parse_mlir_module(full_path, context)
    codegen_fns = backend.get_codegen_implementation(options)
    module_map = backend.get_module_map()
    module = src.make_ir(options, codegen_fns, module_map, context)
    return module


def compile_ir(full_path, kernel_name, save_path):
    options = get_options(full_path)
    stages = get_satges(options)
    metadata = get_metadata(options)
    first_stage = 1
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
