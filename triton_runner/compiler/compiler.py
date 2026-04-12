from pathlib import Path

from triton.compiler.compiler import CompiledKernel, json


class RunnerCompiledKernel(CompiledKernel):

    def _init_handles(self):
        # create launcher – use TVM-FFI driver when enabled and cubin is available (CUDA only)
        from triton_runner import TRITON_RUNNER_ENABLE_TVM_FFI
        if TRITON_RUNNER_ENABLE_TVM_FFI and self.metadata.target.backend == "cuda":
            if self.module is not None:
                return
            from triton_runner.tvm_ffi.driver import TvmFfiLauncher
            self._run = TvmFfiLauncher(self.src, self.metadata, self.asm)
            # TVM-FFI loads the cubin internally; set module to a sentinel to
            # prevent re-initialisation on the next call.
            self.module = True
            self.function = None
            return
        super()._init_handles()

class CompiledTVMFFIKernel:
    def __init__(self, cubin_path, json_path):
        self._cubin_path = cubin_path
        self._json_path = json_path
        self._run_launcher = None
        self._grid_runner_cache = {}

    def _get_launcher(self):
        if self._run_launcher is None:
            from triton_runner.tvm_ffi.driver import TvmFfiLauncher
            cubin_bytes = Path(self._cubin_path).read_bytes()
            metadata = json.loads(Path(self._json_path).read_text())
            self._run_launcher = TvmFfiLauncher(None, metadata, {"cubin": cubin_bytes})
        return self._run_launcher

    def _launch(self, gridX, gridY, gridZ, *args):
        launcher = self._get_launcher()
        runtime_args = launcher._runtime_args(args)
        if launcher._launch_bound_args_for_tvm_ffi is None:
            launcher._tvm_func(launcher._registry_handle, gridX, gridY, gridZ, *runtime_args)
            return
        launcher._launch_bound_args_for_tvm_ffi(gridX, gridY, gridZ, *runtime_args)

    def run(self, gridX, gridY, gridZ, launch_enter_hook, launch_exit_hook, *args):
        self._launch(gridX, gridY, gridZ, *args)

    def __getitem__(self, grid):
        launcher = self._get_launcher()
        is_plain_int_grid = isinstance(grid, tuple) and len(grid) == 3 and all(type(v) is int for v in grid)
        key = grid if is_plain_int_grid else (int(grid[0]), int(grid[1]), int(grid[2]))
        cached = self._grid_runner_cache.get(key)
        if cached is not None:
            return cached
        runner = launcher.get_grid_launcher(*key)
        self._grid_runner_cache[key] = runner
        return runner
