from pathlib import Path

from triton import knobs
from triton.compiler.compiler import CompiledKernel, json
from triton.runtime.driver import driver


class RunnerCompiledKernel(CompiledKernel):

    def _init_handles(self):
        # create launcher – use TVM-FFI driver when enabled, otherwise CudaLauncher
        from triton_runner import TRITON_RUNNER_ENABLE_TVM_FFI
        if TRITON_RUNNER_ENABLE_TVM_FFI:
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

    def run(self, gridX, gridY, gridZ, stream, launch_enter_hook, launch_exit_hook, *args):
        launcher = self._get_launcher()
        launcher(
            gridX,
            gridY,
            gridZ,
            stream,
            None,
            None,
            None,
            launch_enter_hook,
            launch_exit_hook,
            *args,
        )

    def __getitem__(self, grid):
        def runner(*args, stream=None):
            if stream is None:
                device = driver.active.get_current_device()
                stream = driver.active.get_current_stream(device)
            self.run(
                grid[0],
                grid[1],
                grid[2],
                stream,
                knobs.runtime.launch_enter_hook,
                knobs.runtime.launch_exit_hook,
                *args,
            )

        return runner
