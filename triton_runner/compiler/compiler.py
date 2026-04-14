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
    def __init__(self, cubin_bytes, metadata):
        self._cubin_bytes = cubin_bytes
        self._metadata = metadata
        self._run_launcher = None

    def _get_launcher(self):
        if self._run_launcher is None:
            from triton_runner.tvm_ffi.driver import TvmFfiLauncher
            self._run_launcher = TvmFfiLauncher(None, self._metadata, {"cubin": self._cubin_bytes})
        return self._run_launcher

    def _launch(self, gridX, gridY, gridZ, *args):
        self._get_launcher().launch(gridX, gridY, gridZ, *args)

    def run(self, gridX, gridY, gridZ, launch_enter_hook, launch_exit_hook, *args):
        self._launch(gridX, gridY, gridZ, *args)

    def __getitem__(self, grid):
        launcher = self._get_launcher()

        def runner(*args, stream=None):
            launcher.launch(grid[0], grid[1], grid[2], *args)

        return runner
