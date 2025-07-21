from triton.runtime.driver import driver
from triton.runtime.jit import JITFunction, KernelInterface, T
from triton._utils import find_paths_if, get_iterable_path
from typing import Callable, Iterable, Optional, Union, overload
import os


class RunnerJITFunction(JITFunction[KernelInterface[T]]):

    def runner(self, grid, bound_args, kwargs, options, sigkeys, sigvals):
        signature_str = " ".join(sigvals)
        filtered_keys = [k for k in kwargs if k not in options.__dict__ and k not in sigkeys]
        runner_dir_set = {"cubin_dir", "ttir_dir", "ttgir_dir", "llir_dir", "ptx_dir"}
        for k in filtered_keys:
            if k.lower() in runner_dir_set:
                from .jit_utils import jit_launch
                return jit_launch(k[:-4].lower(), kwargs[k], self.__name__, bound_args.values(), signature_str, grid,
                                  options)
            else:
                raise KeyError("Keyword argument %s was specified but unrecognised" % k)

    def run(self, *args, grid, warmup, **kwargs):
        kwargs["debug"] = kwargs.get("debug", self.debug) or os.environ.get("TRITON_DEBUG", "0") == "1"

        # parse options
        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)

        # Execute pre run hooks with args and kwargs
        for hook in self.pre_run_hooks:
            hook(*args, **kwargs)

        kernel_cache, target, backend, binder = self.device_caches[device]
        bound_args, specialization, options = binder(*args, **kwargs)

        # compute cache key
        key = str(specialization) + str(options)
        kernel = kernel_cache.get(key, None)

        # Kernel is not cached; we have to compile.
        if kernel is None:
            # options
            options = backend.parse_options(kwargs)
            # signature
            sigkeys = [x.name for x in self.params]
            sigvals = [x[0] for x in specialization]
            signature = {k: v for (k, v) in zip(sigkeys, sigvals)}
            # check arguments
            assert "device_type" not in kwargs, "device_type option is deprecated; current target will be used"
            assert "device" not in kwargs, "device option is deprecated; current device will be used"
            assert "stream" not in kwargs, "stream option is deprecated; current stream will be used"
            assert grid is not None
            if callable(grid):
                grid = grid(bound_args)
            kernel_launcher = self.runner(grid, bound_args, kwargs, options, sigkeys, sigvals)
            if kernel_launcher is None:
                # constexprs
                constexprs = find_paths_if(sigvals, lambda _, val: val == "constexpr")
                constexprs = {path: get_iterable_path(list(bound_args.values()), path) for path in constexprs}
                # attributes
                attrvals = [x[1] for x in specialization]
                attrs = find_paths_if(attrvals, lambda _, x: isinstance(x, str))
                attrs = {k: backend.parse_attr(get_iterable_path(attrvals, k)) for k in attrs}
                if self._call_hook(key, signature, device, constexprs, options, [attrs], warmup, before=True):
                    return None
                # compile the kernel
                src = self.ASTSource(self, signature, constexprs, attrs)
                kernel = self.compile(src, target=target, options=options.__dict__)
                kernel_cache[key] = kernel
                self._call_hook(key, signature, device, constexprs, options, [attrs], warmup, before=False)
                from .jit_utils import jit_kerel_launch
                kernel_launcher = jit_kerel_launch(kernel, sigvals, bound_args.values(), grid)
        else:
            from .jit_utils import jit_kerel_launch
            sigvals = [x[0] for x in specialization]
            kernel_launcher = jit_kerel_launch(kernel, sigvals, bound_args.values(), grid)

        # Check that used global values have not changed.
        not_present = object()
        for (name, _), (val, globals_dict) in self.used_global_vals.items():
            if (newVal := globals_dict.get(name, not_present)) != val:
                raise RuntimeError(
                    f"Global variable {name} has changed since we compiled this kernel, from {val} to {newVal}")

        if not warmup:
            kernel_launcher.run()

        return kernel_launcher

    def __init__(self, fn, version=None, do_not_specialize=None, do_not_specialize_on_alignment=None, debug=None,
                 noinline=None, repr=None, launch_metadata=None):
        super().__init__(fn, version, do_not_specialize, do_not_specialize_on_alignment, debug, noinline, repr,
                         launch_metadata)


# -----------------------------------------------------------------------------
# jit decorator
# -----------------------------------------------------------------------------


@overload
def jit(fn: T) -> RunnerJITFunction[T]:
    ...


@overload
def jit(
    *,
    version=None,
    repr: Optional[Callable] = None,
    launch_metadata: Optional[Callable] = None,
    do_not_specialize: Optional[Iterable[int]] = None,
    do_not_specialize_on_alignment: Optional[Iterable[int]] = None,
    debug: Optional[bool] = None,
    noinline: Optional[bool] = None,
) -> Callable[[T], RunnerJITFunction[T]]:
    ...


def jit(
    fn: Optional[T] = None,
    *,
    version=None,
    repr: Optional[Callable] = None,
    launch_metadata: Optional[Callable] = None,
    do_not_specialize: Optional[Iterable[int]] = None,
    do_not_specialize_on_alignment: Optional[Iterable[int]] = None,
    debug: Optional[bool] = None,
    noinline: Optional[bool] = None,
) -> Union[RunnerJITFunction[T], Callable[[T], RunnerJITFunction[T]]]:

    def decorator(fn: T) -> RunnerJITFunction[T]:
        assert callable(fn)
        return RunnerJITFunction(
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
