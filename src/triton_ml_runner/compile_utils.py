import triton
import hashlib

def triton_compile(full_name):
    target = triton.runtime.driver.active.get_current_target()
    backend = triton.compiler.make_backend(target)
    context = triton._C.libtriton.ir.context()
    src = triton.compiler.IRSource(full_name, context, backend)
    extra_options = src.parse_options()

    options = backend.parse_options(dict(None or dict(), **extra_options))
    stages = {}
    backend.add_stages(stages, options)

    triton._C.libtriton.ir.load_dialects(context)
    backend.load_dialects(context)

    module = triton._C.libtriton.ir.parse_mlir_module(full_name, context)
    first_stage = 1
    env_vars = triton._C.libtriton.get_cache_invalidating_env_vars()
    key = f"{src.hash()}-{backend.hash()}-{options.hash()}-{str(sorted(env_vars.items()))}"
    hash = "mlir_runner" + hashlib.sha256(key.encode("utf-8")).hexdigest()
    metadata = {
        "hash": hash,
        "target": target,
        **options.__dict__,
        **env_vars,
    }
    metadata["triton_version"] = triton.__version__
    codegen_fns = backend.get_codegen_implementation(options)
    module_map = backend.get_module_map()
    module = src.make_ir(options, codegen_fns, module_map, context)
    for ext, compile_ir in list(stages.items())[first_stage:]:
        next_module = compile_ir(module, metadata)
        module = next_module
    return module, metadata
