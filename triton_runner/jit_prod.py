from triton.runtime.jit import JITFunction, KernelInterface, T, JitFunctionInfo
from triton.runtime.jit import KernelParam, create_function_from_signature, find_paths_if, get_iterable_path, serialize_specialization_data, get_full_name
from triton.runtime.jit import DependenciesFinder, MockTensor, Dict, Tuple, Any, Optional, Callable, Iterable, Union, overload
from triton.runtime import driver
from triton import knobs
from collections import defaultdict
import ast
import inspect
import re
import textwrap
from . import TRITON_RUNNER_PROD_TEST
from .compiler import CompiledTVMFFIKernel

_kernel_cache_dirs: Dict[str, set] = defaultdict(set)


def track_kernel_cache_dir(kernel, name):
    from .color_print import blue_print, red_print
    from triton.runtime.cache import get_cache_manager
    cache_dir = get_cache_manager(kernel.hash).cache_dir
    old_num = len(_kernel_cache_dirs[name])
    _kernel_cache_dirs[name].add(cache_dir)
    if old_num > 0 and old_num != len(_kernel_cache_dirs[name]):
        red_print(f"[ProdJIT] {name} has multiple cache dirs: {_kernel_cache_dirs[name]}")
    # else:
    #     blue_print(f"[ProdJIT] {name} compiled → {cache_dir}")


def update_kernel_metadata(kernel, bound_args, specialization):
    import glob
    import json
    import os
    from triton.runtime.cache import get_cache_manager
    from .version_utils import triton_version
    from . import __version__
    kernel_signature = tuple((k, arg_type, spec) for k, (arg_type, spec) in zip(bound_args.keys(), specialization))
    kernel_cache_dir = get_cache_manager(kernel.hash).cache_dir
    json_files = [f for f in glob.glob(os.path.join(kernel_cache_dir, "*.json")) if not os.path.basename(f).startswith("__grp__")]
    if json_files:
        json_path = json_files[0]
        with open(json_path, 'r') as f:
            meta = json.load(f)
        runner_meta = {
            "kernel_signature": str(kernel_signature),
            "triton_version": triton_version,
            "triton_runner_version": __version__,
            **meta,
        }
        with open(json_path, 'w') as f:
            json.dump(runner_meta, f)
    return runner_meta

class ProdJITFunction(JITFunction[KernelInterface[T]]):

    def run(self, *args, grid, warmup, **kwargs):

        # parse options
        device = driver.active.get_current_device()
        # stream = driver.active.get_current_stream(device)

        # Execute pre run hooks with args and kwargs
        for hook in self.pre_run_hooks:
            hook(*args, **kwargs)

        kernel_cache, target, backend, binder = self.device_caches[device]
        # specialization is list[tuple[str, Any]], where first element of tuple is
        # the type and the second parameter is the 'specialization' value.
        bound_args, specialization, options = binder(*args, **kwargs)

        # compute cache key
        key = str(specialization) + str(options)
        kernel = kernel_cache.get(key, None)

        # Kernel is not cached; we have to compile.
        if kernel is None:
            # options
            options = backend.parse_options(kwargs)
            object.__setattr__(options, "tvm", True)
            # signature
            sigkeys = [x.name for x in self.params]
            sigvals = [x[0] for x in specialization]
            signature = {k: v for (k, v) in zip(sigkeys, sigvals)}
            # check arguments
            assert "device_type" not in kwargs, "device_type option is deprecated; current target will be used"
            assert "device" not in kwargs, "device option is deprecated; current device will be used"
            assert "stream" not in kwargs, "stream option is deprecated; current stream will be used"
            for k in kwargs:
                if k not in options.__dict__ and k not in sigkeys:
                    raise KeyError("Keyword argument %s was specified but unrecognised" % k)
            # constexprs
            constexprs = find_paths_if(sigvals, lambda _, val: val == "constexpr")
            constexprs = {path: get_iterable_path(list(bound_args.values()), path) for path in constexprs}
            # attributes
            attrvals = [x[1] for x in specialization]
            attrs = find_paths_if(attrvals, lambda _, x: isinstance(x, str))
            attrs = {k: backend.parse_attr(get_iterable_path(attrvals, k)) for k in attrs}
            if self._call_hook(knobs.runtime.jit_cache_hook, key, signature, device, constexprs, options, [attrs],
                               warmup):
                return None
            # compile the kernel
            src = self.ASTSource(self, signature, constexprs, attrs)
            kernel = self.compile(src, target=target, options=options.__dict__)
            kernel_cache[key] = kernel
            self._call_hook(knobs.runtime.jit_post_compile_hook, key, signature, device, constexprs, options, [attrs],
                            warmup)
            kernel._init_handles()
            runner_metadata = update_kernel_metadata(kernel, bound_args, specialization)
            tvm_kernel = CompiledTVMFFIKernel(kernel.function, runner_metadata)
            tvm_kernel._get_launcher()
            kernel_cache[key + "_tvm"] = tvm_kernel

        if TRITON_RUNNER_PROD_TEST:
            track_kernel_cache_dir(kernel, self.__name__)

            # Check that used global values have not changed.
            not_present = object()
            for (name, _), (val, globals_dict) in self.used_global_vals.items():
                if (newVal := globals_dict.get(name, not_present)) != val:
                    raise RuntimeError(
                        f"Global variable {name} has changed since we compiled this kernel, from {val} to {newVal}")

        if not warmup:
            # canonicalize grid
            assert grid is not None
            if callable(grid):
                grid = grid(bound_args)
            grid_size = len(grid)
            grid_0 = grid[0]
            grid_1 = grid[1] if grid_size > 1 else 1
            grid_2 = grid[2] if grid_size > 2 else 1
            # launch kernel via TVM-FFI
            tvm_kernel = kernel_cache[key + "_tvm"]
            tvm_kernel.run(grid_0, grid_1, grid_2,
                           knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook,
                           *bound_args.values())
        return kernel


# -----------------------------------------------------------------------------
# `jit` decorator
# -----------------------------------------------------------------------------


@overload
def jit(fn: T) -> JITFunction[T]:
    ...


@overload
def jit(
    *,
    version=None,
    repr: Optional[Callable] = None,
    launch_metadata: Optional[Callable] = None,
    do_not_specialize: Optional[Iterable[int | str]] = None,
    do_not_specialize_on_alignment: Optional[Iterable[int | str]] = None,
    debug: Optional[bool] = None,
    noinline: Optional[bool] = None,
) -> Callable[[T], ProdJITFunction[T]]:
    ...


def jit(
    fn: Optional[T] = None,
    *,
    version=None,
    repr: Optional[Callable] = None,
    launch_metadata: Optional[Callable] = None,
    do_not_specialize: Optional[Iterable[int | str]] = None,
    do_not_specialize_on_alignment: Optional[Iterable[int | str]] = None,
    debug: Optional[bool] = None,
    noinline: Optional[bool] = None,
) -> Union[ProdJITFunction[T], Callable[[T], ProdJITFunction[T]]]:
    """
    Decorator for JIT-compiling a function using the Triton compiler.

    :note: When a jit'd function is called, arguments are
        implicitly converted to pointers if they have a :code:`.data_ptr()` method
        and a `.dtype` attribute.

    :note: This function will be compiled and run on the GPU. It will only have access to:

           * python primitives,
           * builtins within the triton package,
           * arguments to this function,
           * other jit'd functions

    :param fn: the function to be jit-compiled
    :type fn: Callable
    """

    def decorator(fn: T) -> ProdJITFunction[T]:
        assert callable(fn)
        if knobs.runtime.interpret:
            from .interpreter import InterpretedFunction
            return InterpretedFunction(fn, version=version, do_not_specialize=do_not_specialize,
                                       do_not_specialize_on_alignment=do_not_specialize_on_alignment, debug=debug,
                                       noinline=noinline, repr=repr, launch_metadata=launch_metadata)
        else:
            return ProdJITFunction(
                fn,
                version=version,
                do_not_specialize=do_not_specialize,
                do_not_specialize_on_alignment=do_not_specialize_on_alignment,
                debug=debug,
                noinline=noinline,
                repr=repr,
                launch_metadata=launch_metadata,
            )

    if fn is not None:
        return decorator(fn)

    else:
        return decorator
