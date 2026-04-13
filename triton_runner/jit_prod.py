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
    json_files = glob.glob(os.path.join(kernel_cache_dir, "*.json"))
    if json_files:
        json_path = json_files[0]
        with open(json_path, 'r') as f:
            meta = json.load(f)
        meta = {
            "kernel_signature": str(kernel_signature),
            "triton_version": triton_version,
            "triton_runner_version": __version__,
            **meta,
        }
        with open(json_path, 'w') as f:
            json.dump(meta, f)

class ProdJITFunction(JITFunction[KernelInterface[T]]):

    def is_gluon(self):
        return False

    def _call_hook(
        self,
        hook,
        key,
        signature,
        device,
        constants,
        options,
        configs,
        is_warmup,
    ) -> bool | None:
        if not hook:
            return None

        name = self.fn.__qualname__
        module = self.fn.__module__
        arg_reprs = ", ".join([f"{param.name}: {ty}" for param, ty in zip(self.params, key[1])])
        repr = f"{name}[num_warps={options.num_warps}, num_ctas={options.num_ctas}, num_stages={options.num_stages}, enable_fp_fusion={options.enable_fp_fusion}, launch_cooperative_grid={options.launch_cooperative_grid}]({arg_reprs})"
        full_name = get_full_name(self.fn)

        specialization_data = serialize_specialization_data(full_name, signature, constants, configs[0], options, key)

        kwargs = {
            'signature': signature,
            'device': device,
            'constants': constants,
            'num_warps': options.num_warps,
            'num_ctas': options.num_ctas,
            'num_stages': options.num_stages,
            'enable_fp_fusion': options.enable_fp_fusion,
            'launch_cooperative_grid': options.launch_cooperative_grid,
            'extern_libs': options.extern_libs,
            'configs': configs,
            'specialization_data': specialization_data,
            'is_warmup': is_warmup,
        }

        return hook(
            key=key,
            repr=repr,
            fn=JitFunctionInfo(module, name, self),
            compile={"key": key, **kwargs},
            is_manual_warmup=is_warmup,
            already_compiled=False,
        )

    def add_pre_run_hook(self, hook):
        '''
        Add a hook that will be executed prior to the execution of run
        function with args and kwargs passed into the kernel
        '''
        assert callable(hook)
        self.pre_run_hooks.append(hook)

    def create_binder(self):
        """
        Precompute as much as possible.
        """
        from triton.compiler import CompiledKernel, compile, ASTSource, make_backend
        target = driver.active.get_current_target()
        backend = make_backend(target)
        self.CompiledKernel = CompiledKernel
        self.compile = compile
        self.ASTSource = ASTSource
        binder = create_function_from_signature(self.signature, self.params, backend)
        return {}, target, backend, binder

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
            update_kernel_metadata(kernel, bound_args, specialization)
            import glob as _glob
            import os as _os
            import json
            from triton.runtime.cache import get_cache_manager
            _cubin_dir = get_cache_manager(kernel.hash).cache_dir
            _grp_json = _glob.glob(_os.path.join(_cubin_dir, "__grp__*.json"))[0]
            _child_paths = json.load(open(_grp_json))["child_paths"]
            kernel_cache[key + "_tvm"] = CompiledTVMFFIKernel(
                _child_paths[kernel.name + ".cubin"],
                _child_paths[kernel.name + ".json"],
            )

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

    def repr(self, _):
        return self._fn_name if self._repr is None else self._repr(_)

    def __init__(self, fn, version=None, do_not_specialize=None, do_not_specialize_on_alignment=None, debug=None,
                 noinline=None, repr=None, launch_metadata=None):
        do_not_specialize = do_not_specialize if do_not_specialize else []
        do_not_specialize_on_alignment = do_not_specialize_on_alignment if do_not_specialize_on_alignment else []

        self.fn = fn
        self.module = fn.__module__
        self.version = version
        self.signature = inspect.signature(fn)
        self.do_not_specialize = do_not_specialize
        self.do_not_specialize_on_alignment = do_not_specialize_on_alignment
        self.starting_line_number = inspect.getsourcelines(fn)[1]
        self._repr = repr
        self._fn_name = get_full_name(fn)
        self.launch_metadata = launch_metadata

        self.params = []
        for i, param in enumerate(self.signature.parameters.values()):
            dns = i in do_not_specialize or param.name in do_not_specialize
            dns_oa = i in do_not_specialize_on_alignment or param.name in do_not_specialize_on_alignment
            self.params.append(KernelParam(i, param, dns, dns_oa))

        # function source code (without decorators)
        src = textwrap.dedent(inspect.getsource(fn))
        src = src[re.search(r"^def\s+\w+\s*\(", src, re.MULTILINE).start():]
        self._unsafe_update_src(src)
        # cache of just-in-time compiled kernels
        self.device_caches = defaultdict(self.create_binder)
        self.hash = None

        # Map of global variables used by the function and any functions it
        # transitively calls, plus their values.  The values are collected when
        # the function is first compiled.  Then every time we run the function,
        # we check that the values of the globals match what's expected,
        # otherwise we raise an error.
        #
        # Different functions can have different __globals__ maps, so the map
        # key is actually (var name, id(__globals__)), and the map value is
        # (value, __globals__).
        self.used_global_vals: Dict[Tuple[str, int], Tuple[Any, Dict[str, Any]]] = {}

        # JITFunction can be instantiated as kernel
        # when called with a grid using __getitem__
        self.kernel = None
        self.debug = debug
        self.noinline = noinline

        # TODO(jlebar): Remove uses of these fields outside this file, then
        # remove the fields here.
        self.arg_names = [p.name for p in self.params]
        self.constexprs = [p.num for p in self.params if p.is_constexpr]

        # Hooks that will be called prior to executing "run"
        self.pre_run_hooks = []

        # reuse docs of wrapped function
        self.__doc__ = fn.__doc__
        self.__name__ = fn.__name__
        self.__qualname__ = fn.__qualname__
        self.__globals__ = fn.__globals__
        self.__module__ = fn.__module__

    def get_capture_scope(self):
        return self.__globals__ | inspect.getclosurevars(self.fn).nonlocals

    @property
    def cache_key(self):
        # TODO : hash should be attribute of `self`
        if self.hash is None:
            nonlocals = inspect.getclosurevars(self.fn).nonlocals
            dependencies_finder = DependenciesFinder(name=self._fn_name, globals=self.__globals__, nonlocals=nonlocals,
                                                     src=self.src)
            dependencies_finder.visit(self.parse())
            self.hash = dependencies_finder.ret + str(self.starting_line_number)
            self.used_global_vals = dict(sorted(dependencies_finder.used_global_vals.items()))
        return self.hash

    @property
    def type(self):
        from triton.language.core import constexpr
        return constexpr

    def warmup(self, *args, grid, **kwargs):
        return self.run(grid=grid, warmup=True, *map(MockTensor.wrap_dtype, args), **kwargs)

    def preload(self, specialization_data):
        from ..compiler import compile, ASTSource
        import json
        import triton.language as tl
        device = driver.active.get_current_device()
        deserialized_obj = json.loads(specialization_data)
        if deserialized_obj['name'] != self._fn_name:
            raise RuntimeError(
                f"Specialization data is for {deserialized_obj['name']} but trying to preload for {self._fn_name}")
        constant_keys = map(tuple, deserialized_obj['constant_keys'])
        constant_vals = deserialized_obj['constant_vals']
        constants = {
            key: tl.dtype(value) if tl.dtype.is_dtype(value) else value
            for key, value in zip(constant_keys, constant_vals)
        }
        attrs_keys = map(tuple, deserialized_obj['attrs_keys'])
        attrs_vals = deserialized_obj['attrs_vals']
        attrs = dict(zip(attrs_keys, attrs_vals))
        signature = dict(deserialized_obj['signature'].items())
        src = ASTSource(self, signature, constants, attrs)
        options = {
            key: tuple(value) if isinstance(value, list) else value
            for key, value in deserialized_obj['options'].items()
        }
        key = deserialized_obj['key']
        kernel = compile(src, None, options)
        self.device_caches[device][0][key] = kernel
        return kernel

    # we do not parse `src` in the constructor because
    # the user might want to monkey-patch self.src dynamically.
    # Our unit tests do this, for example.
    def parse(self):
        tree = ast.parse(self.src)
        assert isinstance(tree, ast.Module)
        assert len(tree.body) == 1
        assert isinstance(tree.body[0], ast.FunctionDef)
        return tree

    def __call__(self, *args, **kwargs):
        raise RuntimeError("Cannot call @triton.jit'd outside of the scope of a kernel")

    def __setattr__(self, name, value):
        # - when `.src` attribute is set, cache key of all callers need to be re-computed
        if name == "src":
            raise AttributeError(f"Cannot set attribute '{name}' directly. "
                                 f"Use '_unsafe_update_src()' and manually clear `.hash` of all callers"
                                 f"instead.")
        super(ProdJITFunction, self).__setattr__(name, value)

    def _unsafe_update_src(self, new_src):
        """
        The only method allowed to modify src.
        Bypasses the __setattr__ restriction by calling super().__setattr__ directly.
        """
        self.hash = None
        object.__setattr__(self, 'src', new_src)

    def __repr__(self):
        return f"JITFunction({self.module}:{self.fn.__qualname__})"

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