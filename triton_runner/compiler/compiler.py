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

    def _launch(self, gridX, gridY, gridZ, launch_enter_hook, launch_exit_hook, *args):
        self._get_launcher().launch(gridX, gridY, gridZ, launch_enter_hook, launch_exit_hook, *args)

    def run(self, gridX, gridY, gridZ, launch_enter_hook, launch_exit_hook, *args):
        self._launch(gridX, gridY, gridZ, launch_enter_hook, launch_exit_hook, *args)

    def __getitem__(self, grid):
        from triton import knobs

        def runner(*args, stream=None):
            self._launch(grid[0], grid[1], grid[2],
                         knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook, *args)

        return runner
