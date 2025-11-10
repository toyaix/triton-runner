import triton
from triton.runtime.driver import driver
from triton.runtime.jit import JITFunction, KernelInterface, T
from typing import Callable, Iterable, Optional, Union, overload
from .compiler import native_compile, get_source_ir
import os
import json
import re
from .dump_utils import get_injected_ir


class RunnerJITFunction(JITFunction[KernelInterface[T]]):

    def get_runner_args_set(self):
        return {"ttir_dir", "ttgir_dir", "llir_dir", "ptx_dir", "cubin_dir"}

    def get_dump_args_set(self):
        return {"dump_tensor", "dump_value", "dump_grid"}

    def is_python_dump(self, kwargs, source_dir_type):
        return self.need_dump(kwargs) and source_dir_type not in ["ttir_dir", "ttgir_dir"]

    def get_source_dir_type(self, need_check_lst):
        runner_args_set = self.get_runner_args_set()
        dump_args_set = self.get_dump_args_set()
        for k in need_check_lst:
            if not k in runner_args_set | dump_args_set:
                raise KeyError("Keyword argument %s was specified but unrecognised" % k)
        for k in need_check_lst:
            if k in runner_args_set:
                return k

    def get_runner_source_dir_str(self, kwargs):
        runner_args_set = self.get_runner_args_set()
        for k in kwargs:
            if k in runner_args_set:
                return kwargs[k] + f"/{self.__name__}.{k[:-4]}"
        return ""

    def _pack_args(self, backend, kwargs, bound_args, specialization, options):
        from triton._utils import find_paths_if, get_iterable_path
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
        # for k in kwargs:
        #     if k not in options.__dict__ and k not in sigkeys:
        #         raise KeyError("Keyword argument %s was specified but unrecognised" % k)
        source_dir_type = self.get_source_dir_type(kwargs, options, sigkeys)
        # constexprs
        constexprs = find_paths_if(sigvals, lambda _, val: val == "constexpr")
        constexprs = {path: get_iterable_path(list(bound_args.values()), path) for path in constexprs}
        # attributes
        attrvals = [x[1] for x in specialization]
        attrs = find_paths_if(attrvals, lambda _, x: isinstance(x, str))
        attrs = {k: backend.parse_attr(get_iterable_path(attrvals, k)) for k in attrs}

        return options, signature, constexprs, attrs, source_dir_type

    def _do_compile(self, key, signature, device, constexprs, options, attrs, warmup, source_dir_type, kwargs, bound_args=None):
        from triton import knobs

        kernel_cache, _, target, backend, _ = self.device_caches[device]

        # [Triton Runner] dump before _call_hook
        src = self.get_src_and_save_dump_file(kwargs, source_dir_type, signature, constexprs, attrs, target, options, bound_args)
        if self._call_hook(knobs.runtime.jit_cache_hook, key, signature, device, constexprs, options, [attrs], warmup):
            return None
        ast_src = self.ASTSource(self, signature, constexprs, attrs)

        # [Triton Runner] dump after _call_hook
        src, metadata_json = self.get_src_and_metadata_json(kwargs, source_dir_type, src, ast_src)
        # TODO: don't support _async_compile
        # async_mode = _async_compile.active_mode.get()
        # if async_mode is not None:

        #     env_vars = get_cache_invalidating_env_vars()
        #     cache_key = get_cache_key(src, backend, options, env_vars)

        #     def async_compile():
        #         return self.compile(src, target=target, options=options.__dict__, _env_vars=env_vars)

        #     def finalize_compile(kernel):
        #         kernel_cache[key] = kernel
        #         self._call_hook(knobs.runtime.jit_post_compile_hook, key, signature, device, constexprs, options,
        #                         [attrs], warmup)

        #     kernel = async_mode.submit(cache_key, async_compile, finalize_compile)
        # else:
        source_path = self.__dict__["__globals__"]["__file__"]
        # [Triton Runner] change compile
        kernel = native_compile(src, ast_src, metadata_json, target=target, options=options.__dict__, source_path=source_path)
        # kernel = self.compile(src, target=target, options=options.__dict__)
        kernel_cache[key] = kernel
        self._call_hook(knobs.runtime.jit_post_compile_hook, key, signature, device, constexprs, options, [attrs],
                        warmup)
        return kernel

    def need_dump(self, kwargs):
        return "dump_tensor" in kwargs
        # return "dump_tensor" in kwargs and "dump_value" in kwargs and "ttir_dir" in kwargs

    def insert_dump_tensor_param(self, full_text):
        pattern = re.compile(r'(tt\.func\s+public\s+@\w+\s*)\((.*?)\)(\s*attributes\s*{[^}]*}\s*{)', re.DOTALL)

        def replacer(match):
            prefix, args_str, suffix = match.groups()
            new_args_str = args_str + ', %runner_dump_tensor: !tt.ptr<f32>'
            return f"{prefix}({new_args_str}){suffix}"

        return pattern.sub(replacer, full_text, count=1)

    def inject_ssa_ir_dump_store(self, full_text, ssa_value, dump_grid):
        pattern = re.compile(
            rf'^(?P<indent>\s*){ssa_value}\s*=\s*'
            r'(?P<op>\S+)\s+'
            r'.*'
            r'tensor<'
            r'(?P<size>(?:\d+x)*\d+)'
            r'(?:x(?:[^,<>]|<[^>]*>)+)?'
            r'(?:,\s*(?P<encoding>#[^>]+))?'
            r'>'
            r'[^<]*?'
            r'loc\((?P<loc>#[^)]+)\)',
            re.MULTILINE
        )
        def make_replacer(dump_grid):
            def replacer(match):
                original_line = match.group(0)
                indent = match.group("indent")
                op = match.group("op")
                size = match.group("size")
                loc = match.group("loc")
                encoding = match.group("encoding")
                return get_injected_ir(ssa_value, op, original_line, indent, size, encoding, loc, dump_grid=dump_grid)
            return replacer
        return pattern.sub(make_replacer(dump_grid), full_text, count=1)

    def inject_dump_op_dump_store(self, full_text):
        ssa_value = r'%\d+'
        pattern = re.compile(
            rf'^(?P<indent>[ \t]*)'
            rf'(?P<ssa_value>{ssa_value})\s*=\s*'
            r'(?P<op>\S+)\s+'
            r'(?P<args>[^\n{}]*)'
            r'\{(?P<attrs>[^\n}]*\btt\.dump\s*=\s*[^\n}]+)\}\s*'
            r':\s*tensor<(?P<size>(?:\d+x)*\d+)(?:x[^\n>]*)?>'
            r'[^\n]*?loc\((?P<loc>#[^)]+)\)'
            r'\n.*=\s*(?P<offset_val>.*)',
            re.MULTILINE
        )
        def make_replacer(replace_id):
            def replacer(match):
                original_line = match.group(0).split('\n')[0]
                indent = match.group("indent")
                op = match.group("op")
                size = match.group("size")
                loc = match.group("loc")
                ssa_value = match.group("ssa_value")
                offset_val = match.group("offset_val")
                clean_line = re.sub(r"\s*\{[^{}]*\}", "", original_line)
                return get_injected_ir(ssa_value, op, clean_line, indent, size, None, loc,
                                       python_dump=True, offset_val=offset_val, replace_id=replace_id)
            return replacer
        full_text, count = pattern.subn(make_replacer(0), full_text, count=1)
        replace_id = 0
        while count > 0:
            replace_id = replace_id + 1
            full_text, count = pattern.subn(make_replacer(replace_id), full_text, count=1)
        return full_text

    def get_src_and_save_dump_file(self, kwargs, source_dir_type, signature, constexprs, attrs, target, options, bound_args):
        src = None
        if self.need_dump(kwargs):
            dump_tensor = kwargs["dump_tensor"]
            from .color_print import check_dump_tensor_dtype
            check_dump_tensor_dtype(dump_tensor)
            if self.is_python_dump(kwargs, source_dir_type):
                src = self.ASTSource(self, signature, constexprs, attrs)
                module = get_source_ir(src, target=target, options=options.__dict__)
                if triton.__version__ in ["3.4.0", "3.5.0"]:
                    from triton import knobs
                    runner_cache_dir = os.path.join(knobs.cache.dir, "runner_cache")
                else:
                    from triton.runtime.cache import default_cache_dir
                    cache_dir = os.getenv("TRITON_CACHE_DIR", "").strip() or default_cache_dir()
                    runner_cache_dir = os.path.join(cache_dir, "runner_cache")
                os.makedirs(runner_cache_dir, exist_ok=True)
                dump_content = self.insert_dump_tensor_param(str(module))
                dump_content = self.inject_dump_op_dump_store(dump_content)
                src = os.path.join(runner_cache_dir, f"{self.__name__}-dump.ttir")
                with open(src, "w") as file:
                    file.write(dump_content)
            signature["dump_tensor"], bound_args["dump_tensor"] = "*fp32", dump_tensor
        return src

    def get_dump_key(self, key, kwargs):
        if "dump_value" in kwargs:
            key += kwargs["dump_value"]
        if (runner_source_dir_str := self.get_runner_source_dir_str(kwargs)):
            key += runner_source_dir_str
        return key

    def get_src_and_metadata_json(self, kwargs, source_dir_type, src, ast_src):
        if source_dir_type:
            source_file_name = f"{self.__name__}.{source_dir_type[:-4]}"
            src = os.path.join(kwargs[source_dir_type], source_file_name)
            if self.need_dump(kwargs) and source_dir_type in ["ttir_dir", "ttgir_dir"]:
                if not os.path.exists(src):
                    src = os.path.join(kwargs[source_dir_type], source_file_name[:-4] + "source")
                if not os.path.exists(src):
                    raise RuntimeError("Check .source/.ttir/.ttgir file for dump.")
                dump_content = self.insert_dump_tensor_param(open(src, "r").read())
                dump_content = self.inject_ssa_ir_dump_store(dump_content, kwargs["dump_value"], kwargs.get("dump_grid", 0))
                src = os.path.join(kwargs[source_dir_type], f"dump.{source_dir_type[:-4]}")
                with open(src, "w") as file:
                    file.write(dump_content)
            metadata_json = {}
            if source_dir_type in {"cubin_dir", "llir_dir", "ptx_dir"}:
                json_file_name = f"{self.__name__}.json"
                json_path = os.path.join(kwargs[source_dir_type], json_file_name)
                metadata_json = json.loads(open(json_path, "r").read())
            return src, metadata_json
        elif self.need_dump(kwargs):
            return src, {}
        return ast_src, {}


class RunnerJITFunctionV3_5_0(RunnerJITFunction[KernelInterface[T]]):

    def get_source_dir_type(self, kwargs, options, sigkeys):
        return super().get_source_dir_type(
            [k.lower() for k in kwargs if k not in options.__dict__ and k not in sigkeys])

    def run(self, *args, grid, warmup, **kwargs):
        from triton import knobs
        from triton.runtime.jit import compute_cache_key

        kwargs["debug"] = kwargs.get("debug", self.debug) or knobs.runtime.debug

        # parse options
        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)

        # Execute pre run hooks with args and kwargs
        for hook in self.pre_run_hooks:
            hook(*args, **kwargs)

        kernel_cache, kernel_key_cache, target, backend, binder = self.device_caches[device]
        # specialization is list[tuple[str, Any]], where first element of tuple is
        # the type and the second parameter is the 'specialization' value.
        bound_args, specialization, options = binder(*args, **kwargs)

        key = compute_cache_key(kernel_key_cache, specialization, options)
        # [Triton Runner] dump key
        key = self.get_dump_key(key, kwargs)
        kernel = kernel_cache.get(key, None)

        # Kernel is not cached; we have to compile.
        if kernel is None:
            options, signature, constexprs, attrs, source_dir_type = self._pack_args(
                backend, kwargs, bound_args, specialization, options)
            kernel = self._do_compile(key, signature, device, constexprs, options, attrs, warmup, source_dir_type, kwargs, bound_args)
            if kernel is None:
                return None

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
            if hasattr(kernel, "result"):
                kernel = kernel.result()
            # launch kernel
            launch_metadata = kernel.launch_metadata(grid, stream, *bound_args.values())
            kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata, launch_metadata,
                       knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook, *bound_args.values())
        return kernel


class RunnerJITFunctionV3_4_0(RunnerJITFunction[KernelInterface[T]]):

    def get_source_dir_type(self, kwargs, options, sigkeys):
        return super().get_source_dir_type(
            [k.lower() for k in kwargs if k not in options.__dict__ and k not in sigkeys])

    def run(self, *args, grid, warmup, **kwargs):
        from triton import knobs
        from triton._utils import find_paths_if, get_iterable_path

        kwargs["debug"] = kwargs.get("debug", self.debug) or knobs.runtime.debug

        # parse options
        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)

        # Execute pre run hooks with args and kwargs
        for hook in self.pre_run_hooks:
            hook(*args, **kwargs)

        kernel_cache, target, backend, binder = self.device_caches[device]
        # specialization is list[tuple[str, Any]], where first element of tuple is
        # the type and the second parameter is the 'specialization' value.
        bound_args, specialization, options = binder(*args, **kwargs)

        # compute cache key
        key = str(specialization) + str(options)
        # [Triton Runner] dump key
        key = self.get_dump_key(key, kwargs)
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

            # check keyword argument and get source_dir_type
            source_dir_type = self.get_source_dir_type(kwargs, options, sigkeys)

            # constexprs
            constexprs = find_paths_if(sigvals, lambda _, val: val == "constexpr")
            constexprs = {path: get_iterable_path(list(bound_args.values()), path) for path in constexprs}
            # attributes
            attrvals = [x[1] for x in specialization]
            attrs = find_paths_if(attrvals, lambda _, x: isinstance(x, str))
            attrs = {k: backend.parse_attr(get_iterable_path(attrvals, k)) for k in attrs}

            # [Triton Runner] dump before _call_hook
            src = self.get_src_and_save_dump_file(kwargs, source_dir_type, signature, constexprs, attrs, target, options, bound_args)
            if self._call_hook(knobs.runtime.jit_cache_hook, key, signature, device, constexprs, options, [attrs],
                               warmup):
                return None
            # compile the kernel
            ast_src = self.ASTSource(self, signature, constexprs, attrs)
            # [Triton Runner] dump after _call_hook
            src, metadata_json = self.get_src_and_metadata_json(kwargs, source_dir_type, src, ast_src)
            kernel_signature = tuple((key, arg_type, spec) for key, (arg_type, spec) in zip(bound_args.keys(), specialization))
            kernel = native_compile(src, ast_src, metadata_json, target=target, options=options.__dict__, kernel_signature=kernel_signature)
            kernel_cache[key] = kernel
            self._call_hook(knobs.runtime.jit_post_compile_hook, key, signature, device, constexprs, options, [attrs],
                            warmup)

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
            # launch kernel
            launch_metadata = kernel.launch_metadata(grid, stream, *bound_args.values())
            kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata, launch_metadata,
                       knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook, *bound_args.values())
        return kernel


class RunnerJITFunction_TLX(RunnerJITFunction[KernelInterface[T]]):

    def get_source_dir_type(self, kwargs, options, sigkeys):
        return super().get_source_dir_type(
            [k.lower() for k in kwargs if k not in options.__dict__ and k not in sigkeys])

    def run(self, *args, grid, warmup, **kwargs):
        from triton import knobs
        from triton.runtime.jit import compute_cache_key

        kwargs["debug"] = kwargs.get("debug", self.debug) or knobs.runtime.debug

        # parse options
        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)

        # Execute pre run hooks with args and kwargs
        for hook in self.pre_run_hooks:
            hook(*args, **kwargs)

        kernel_cache, kernel_key_cache, target, backend, binder = self.device_caches[device]
        # specialization is list[tuple[str, Any]], where first element of tuple is
        # the type and the second parameter is the 'specialization' value.
        bound_args, specialization, options = binder(*args, **kwargs)

        key = compute_cache_key(kernel_key_cache, specialization, options)
        kernel = kernel_cache.get(key, None)

        # Kernel is not cached; we have to compile.
        if kernel is None:
            options, signature, constexprs, attrs, source_dir_type = self._pack_args(
                backend, kwargs, bound_args, specialization, options)
            kernel = self._do_compile(key, signature, device, constexprs, options, attrs, warmup, source_dir_type, kwargs)
            if kernel is None:
                return None

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
            if hasattr(kernel, "result"):
                kernel = kernel.result()
            # launch kernel
            launch_metadata = kernel.launch_metadata(grid, stream, *bound_args.values())
            kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata, launch_metadata,
                       knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook, *bound_args.values())
        return kernel


class RunnerJITFunctionV3_3_x(RunnerJITFunction[KernelInterface[T]]):

    def get_source_dir_type(self, kwargs, options, sigkeys):
        return super().get_source_dir_type(
            [k.lower() for k in kwargs if k not in options.__dict__ and k not in sigkeys])

    def run(self, *args, grid, warmup, **kwargs):
        from triton._utils import find_paths_if, get_iterable_path

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
        # [Triton Runner] dump key
        key = self.get_dump_key(key, kwargs)
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

            # check keyword argument and get source_dir_type
            source_dir_type = self.get_source_dir_type(kwargs, options, sigkeys)

            # constexprs
            constexprs = find_paths_if(sigvals, lambda _, val: val == "constexpr")
            constexprs = {path: get_iterable_path(list(bound_args.values()), path) for path in constexprs}
            # attributes
            attrvals = [x[1] for x in specialization]
            attrs = find_paths_if(attrvals, lambda _, x: isinstance(x, str))
            attrs = {k: backend.parse_attr(get_iterable_path(attrvals, k)) for k in attrs}

            # [Triton Runner] dump before _call_hook
            src = self.get_src_and_save_dump_file(kwargs, source_dir_type, signature, constexprs, attrs, target, options, bound_args)

            if self._call_hook(key, signature, device, constexprs, options, [attrs], warmup, before=True):
                return None
            # compile the kernel
            ast_src = self.ASTSource(self, signature, constexprs, attrs)
            # [Triton Runner] dump after _call_hook
            src, metadata_json = self.get_src_and_metadata_json(kwargs, source_dir_type, src, ast_src)

            kernel = native_compile(src, ast_src, metadata_json, target=target, options=options.__dict__)
            kernel_cache[key] = kernel
            self._call_hook(key, signature, device, constexprs, options, [attrs], warmup, before=False)

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
            # launch kernel
            launch_metadata = kernel.launch_metadata(grid, stream, *bound_args.values())
            kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata,
                       launch_metadata, self.CompiledKernel.launch_enter_hook, self.CompiledKernel.launch_exit_hook,
                       *bound_args.values())
        return kernel

    def __init__(self, fn, version=None, do_not_specialize=None, do_not_specialize_on_alignment=None, debug=None,
                 noinline=None, repr=None, launch_metadata=None):
        super().__init__(fn, version, do_not_specialize, do_not_specialize_on_alignment, debug, noinline, repr,
                         launch_metadata)


class RunnerJITFunctionV3_2_0(RunnerJITFunction[KernelInterface[T]]):

    def get_source_dir_type(self, excess_kwargs, options):
        return super().get_source_dir_type([k.lower() for k in excess_kwargs if k not in options.__dict__])

    def run(self, *args, grid, warmup, **kwargs):
        kwargs["debug"] = kwargs.get("debug", False) or os.environ.get("TRITON_DEBUG", "0") == "1"

        # parse options
        from triton.compiler import make_backend
        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)
        target = driver.active.get_current_target()
        backend = make_backend(target)

        # Execute pre run hooks with args and kwargs
        for hook in self.pre_run_hooks:
            hook(*args, **kwargs)

        if self.binder is None:
            self.create_binder(backend)

        bound_args, sig_and_spec, constexpr_vals, non_constexpr_vals, excess_kwargs = self.binder(*args, **kwargs)

        # compute cache key
        key = ''.join(sig_and_spec) + str((constexpr_vals, excess_kwargs))
        kernel = self.cache[device].get(key, None)

        if kernel is None:
            # Kernel is not cached; we have to compile.
            options = backend.parse_options(kwargs)

            # deprecated arguments
            assert "device_type" not in kwargs, "device_type option is deprecated; current target will be used"
            assert "device" not in kwargs, "device option is deprecated; current device will be used"
            assert "stream" not in kwargs, "stream option is deprecated; current stream will be used"

            # check keyword argument and get source_dir_type
            source_dir_type = self.get_source_dir_type(excess_kwargs, options)

            bound_vals = tuple(bound_args.values())

            # `None` is nullptr. Implicitly convert to *i8. This needs to be
            # done here rather than when we build the signature as otherwise
            # the kernel cache key could not distinguish between byte pointers
            # and None arguments, resulting in a downstream mismatch:
            sigkeys = [self.params[i].name for i in self.non_constexpr_indices]
            sigvals = sig_and_spec[:len(sigkeys)]
            signature = {k: ('*i8' if (v == 'none') else v) for (k, v) in zip(sigkeys, sigvals)}

            configs = (backend.get_attrs_descriptor(self.params, bound_vals), )
            constant_params = configs[0].get_constants()
            constants = {
                p.name: v
                for (v, p) in zip(bound_vals, self.params)
                if p.is_constexpr or (p.num in constant_params) or v is None
            }
            for i, arg in constants.items():
                if callable(arg):
                    raise TypeError(f"Callable constexpr at index {i} is not supported")

            if self._call_hook(key, signature, device, constants, options, configs, warmup, before=True):
                return None
            # compile the kernel
            ast_src = self.ASTSource(self, signature, constants, configs[0])
            metadata_json = {}
            if source_dir_type:
                source_file_name = f"{self.__name__}.{source_dir_type[:-4]}"
                src = os.path.join(kwargs[source_dir_type], source_file_name)
                if source_dir_type in {"cubin_dir", "llir_dir", "ptx_dir"}:
                    json_file_name = f"{self.__name__}.json"
                    json_path = os.path.join(kwargs[source_dir_type], json_file_name)
                    metadata_json = json.loads(open(json_path, "r").read())
            else:
                src = ast_src

            kernel = native_compile(src, ast_src, metadata_json, target=target, options=options.__dict__)
            self.cache[device][key] = kernel
            self._call_hook(key, signature, device, constants, options, configs, warmup, before=False)

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
                # Arguments are passed as a dict to `grid`, by contract.
                # TODO(jlebar): In the new launch API, pass the compiler flags as a
                # second parameter to `grid`.
                grid = grid(bound_args)
            grid_size = len(grid)
            grid_0 = grid[0]
            grid_1 = grid[1] if grid_size > 1 else 1
            grid_2 = grid[2] if grid_size > 2 else 1

            # launch kernel
            launch_metadata = kernel.launch_metadata(grid, stream, *non_constexpr_vals)
            kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata, launch_metadata,
                       self.CompiledKernel.launch_enter_hook, self.CompiledKernel.launch_exit_hook, *non_constexpr_vals)
        return kernel


class RunnerJITFunctionV3_1_0(RunnerJITFunction[KernelInterface[T]]):

    def get_source_dir_type(self, excess_kwargs, options):
        return super().get_source_dir_type([k.lower() for k in excess_kwargs if k not in options.__dict__])

    def run(self, *args, grid, warmup, **kwargs):
        # parse options
        device = driver.active.get_current_device()
        stream = driver.active.get_current_stream(device)
        kwargs["debug"] = self.debug

        # Execute pre run hooks with args and kwargs
        for hook in self.pre_run_hooks:
            hook(*args, **kwargs)

        if self.binder is None:
            self.create_binder()

        bound_args, sig_and_spec, constexpr_vals, non_constexpr_vals, excess_kwargs = self.binder(*args, **kwargs)

        # compute cache key
        key = ''.join(sig_and_spec) + str((constexpr_vals, excess_kwargs))
        kernel = self.cache[device].get(key, None)

        if kernel is None:
            # Kernel is not cached; we have to compile.
            target = driver.active.get_current_target()
            backend = self.make_backend(target)
            options = backend.parse_options(kwargs)

            # deprecated arguments
            assert "device_type" not in kwargs, "device_type option is deprecated; current target will be used"
            assert "device" not in kwargs, "device option is deprecated; current device will be used"
            assert "stream" not in kwargs, "stream option is deprecated; current stream will be used"

            # check keyword argument and get source_dir_type
            source_dir_type = self.get_source_dir_type(excess_kwargs, options)

            bound_vals = tuple(bound_args.values())

            # `None` is nullptr. Implicitly convert to *i8. This needs to be
            # done here rather than when we build the signature as otherwise
            # the kernel cache key could not distinguish between byte pointers
            # and None arguments, resulting in a downstream mismatch:
            sigkeys = [self.params[i].name for i in self.non_constexpr_indices]
            sigvals = sig_and_spec[:len(sigkeys)]
            signature = {k: ('*i8' if (v == 'none') else v) for (k, v) in zip(sigkeys, sigvals)}

            configs = (self._get_config(*bound_vals), )
            constants = {
                p.name: v
                for (v, p) in zip(bound_vals, self.params)
                if p.is_constexpr or p.num in configs[0].equal_to_1 or v is None
            }
            for i, arg in constants.items():
                if callable(arg):
                    raise TypeError(f"Callable constexpr at index {i} is not supported")

            if self._call_hook(key, signature, device, constants, options, configs):
                return None
            # compile the kernel
            ast_src = self.ASTSource(self, signature, constants, configs[0])
            metadata_json = {}
            if source_dir_type:
                source_file_name = f"{self.__name__}.{source_dir_type[:-4]}"
                src = os.path.join(kwargs[source_dir_type], source_file_name)
                if source_dir_type in {"cubin_dir", "llir_dir", "ptx_dir"}:
                    json_file_name = f"{self.__name__}.json"
                    json_path = os.path.join(kwargs[source_dir_type], json_file_name)
                    metadata_json = json.loads(open(json_path, "r").read())
            else:
                src = ast_src
            kernel = native_compile(src, ast_src, metadata_json, target=target, options=options.__dict__)
            self.cache[device][key] = kernel

        # Check that used global values have not changed.
        not_present = object()
        for (name, globals_dict_id), (val, globals_dict) in self.used_global_vals.items():
            if (newVal := globals_dict.get(name, not_present)) != val:
                raise RuntimeError(
                    f"Global variable {name} has changed since we compiled this kernel, from {val} to {newVal}")

        if not warmup:
            # canonicalize grid
            assert grid is not None
            if callable(grid):
                # Arguments are passed as a dict to `grid`, by contract.
                # TODO(jlebar): In the new launch API, pass the compiler flags as a
                # second parameter to `grid`.
                grid = grid(bound_args)
            grid_size = len(grid)
            grid_0 = grid[0]
            grid_1 = grid[1] if grid_size > 1 else 1
            grid_2 = grid[2] if grid_size > 2 else 1

            # launch kernel
            launch_metadata = kernel.launch_metadata(grid, stream, *non_constexpr_vals)
            kernel.run(grid_0, grid_1, grid_2, stream, kernel.function, kernel.packed_metadata, launch_metadata,
                       self.CompiledKernel.launch_enter_hook, self.CompiledKernel.launch_exit_hook, *non_constexpr_vals)
        return kernel
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
        from .tlx_utils import is_tlx
        if is_tlx:
            return RunnerJITFunction_TLX(
                fn,
                version=version,
                do_not_specialize=do_not_specialize,
                do_not_specialize_on_alignment=do_not_specialize_on_alignment,
                debug=debug,
                noinline=noinline,
                repr=repr,
                launch_metadata=launch_metadata,
            )
        elif triton.__version__ == "3.5.0":
            return RunnerJITFunctionV3_5_0(
                fn,
                version=version,
                do_not_specialize=do_not_specialize,
                do_not_specialize_on_alignment=do_not_specialize_on_alignment,
                debug=debug,
                noinline=noinline,
                repr=repr,
                launch_metadata=launch_metadata,
            )
        elif triton.__version__ == "3.4.0":
            return RunnerJITFunctionV3_4_0(
                fn,
                version=version,
                do_not_specialize=do_not_specialize,
                do_not_specialize_on_alignment=do_not_specialize_on_alignment,
                debug=debug,
                noinline=noinline,
                repr=repr,
                launch_metadata=launch_metadata,
            )
        elif triton.__version__ in ["3.3.0", "3.3.1"]:
            return RunnerJITFunctionV3_3_x(
                fn,
                version=version,
                do_not_specialize=do_not_specialize,
                do_not_specialize_on_alignment=do_not_specialize_on_alignment,
                debug=debug,
                noinline=noinline,
                repr=repr,
                launch_metadata=launch_metadata,
            )
        elif triton.__version__ == "3.2.0":
            return RunnerJITFunctionV3_2_0(
                fn,
                version=version,
                do_not_specialize=do_not_specialize,
                do_not_specialize_on_alignment=do_not_specialize_on_alignment,
                debug=debug,
                noinline=noinline,
                repr=repr,
                launch_metadata=launch_metadata,
            )
        elif triton.__version__ in ["3.1.0", "3.0.0"]:
            return RunnerJITFunctionV3_1_0(
                fn,
                version=version,
                do_not_specialize=do_not_specialize,
                debug=debug,
                noinline=noinline,
                repr=repr,
                launch_metadata=launch_metadata,
            )
        else:
            raise RuntimeError(f"Can't support Triton v{triton.__version__}.")

    if fn is not None:
        return decorator(fn)
    else:
        return decorator
