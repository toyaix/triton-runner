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
        grid_x = grid[0]
        grid_y = grid[1]
        grid_z = grid[2]

        if launcher._launch_bound_args_for_tvm_ffi is None:
            runtime_arg_count = len(launcher._runtime_signature)
            signature_arg_count = len(launcher._signature)
            registry_handle = launcher._registry_handle
            tvm_func = launcher._tvm_func
            runtime_arg_indices = launcher._runtime_arg_indices

            if runtime_arg_count == signature_arg_count:
                def runner(*args, stream=None):
                    tvm_func(registry_handle, grid_x, grid_y, grid_z, *args)
            else:
                def runner(*args, stream=None):
                    if len(args) == runtime_arg_count:
                        tvm_func(registry_handle, grid_x, grid_y, grid_z, *args)
                        return
                    if len(args) == signature_arg_count:
                        runtime_args = tuple(args[i] for i in runtime_arg_indices)
                        tvm_func(registry_handle, grid_x, grid_y, grid_z, *runtime_args)
                        return
                    raise ValueError(
                        f"Expected either {runtime_arg_count} runtime args or {signature_arg_count} bound args, got {len(args)}."
                    )
            return runner

        def runner(*args, stream=None):
            launcher.launch(grid_x, grid_y, grid_z, *args)

        return runner
