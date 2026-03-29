from pathlib import Path
import copy

from triton import knobs
from triton.compiler.compiler import CompiledKernel, max_shared_mem, json
from triton.runtime.driver import driver
from triton.runtime.autotuner import OutOfResources
import functools


def _raise_error(err, *args, **kwargs):
    raise copy.deepcopy(err)


class CompiledKernel_v3_5_0(CompiledKernel):

    def _init_handles(self):
        if self.module is not None:
            return

        def raise_(err):
            # clone the exception object so that the one saved in the closure
            # of the partial function below doesn't get assigned a stack trace
            # after the subsequent raise. otherwise, the CompiledKernel instance
            # saved in the (global) kernel cache will keep references to all the
            # locals in the traceback via the exception instance in the closure.
            cloned_err = copy.deepcopy(err)
            self._run = functools.partial(_raise_error, cloned_err)
            raise err

        device = driver.active.get_current_device()
        # create launcher
        # self._run = driver.active.launcher_cls(self.src, self.metadata)

        # create launcher – use TVM-FFI driver when enabled, otherwise CudaLauncher
        from triton_runner import TRITON_TVM_FFI
        if TRITON_TVM_FFI:
            from triton_runner.driver.tvm_ffi_driver import TvmFfiLauncher
            self._run = TvmFfiLauncher(self.src, self.metadata, self.asm)
            # TVM-FFI loads the cubin internally; set module to a sentinel to
            # prevent re-initialisation on the next call.
            self.module = True
            self.function = None
            return
        from triton_runner.driver.v3_5_0.driver import CudaLauncher as CudaLauncher_v3_5_0
        self._run = CudaLauncher_v3_5_0(self.src, self.metadata)
        # not enough shared memory to run the kernel
        max_shared = max_shared_mem(device)
        if self.metadata.shared > max_shared:
            raise_(OutOfResources(self.metadata.shared, max_shared, "shared memory"))
        if hasattr(self.metadata, "tmem_size") and self.metadata.tmem_size is not None:
            # Use blackwell max tmem size for now, this should be moved in device properties
            max_tmem_size = 512  # tmem size in number of columns
            if self.metadata.tmem_size > max_tmem_size:
                raise_(OutOfResources(self.metadata.tmem_size, max_tmem_size, "tensor memory"))
        # if knobs.runtime.kernel_load_start_hook is not None:
        #     knobs.runtime.kernel_load_start_hook(self.module, self.function, self.name, self.metadata_group, self.hash)
        # TODO: n_regs, n_spills should be metadata generated when calling `ptxas`
        self.module, self.function, self.n_regs, self.n_spills, self.n_max_threads = driver.active.utils.load_binary(
            self.name, self.kernel, self.metadata.shared, device)
        warp_size = driver.active.get_current_target().warp_size
        if self.metadata.num_warps * warp_size > self.n_max_threads:
            raise_(OutOfResources(self.metadata.num_warps * warp_size, self.n_max_threads, "threads"))
        from ..version_utils import is_triton_v3_5
        if is_triton_v3_5:
            if knobs.runtime.kernel_load_end_hook is not None:
                knobs.runtime.kernel_load_end_hook(self.module, self.function, self.name, self.metadata_group, self.hash)


class CompiledTVMFFIKernel:
    def __init__(self, cubin_path, json_path):
        self._cubin_path = cubin_path
        self._json_path = json_path
        self._run_launcher = None

    def _get_launcher(self):
        if self._run_launcher is None:
            from triton_runner.driver.tvm_ffi_driver import TvmFfiLauncher
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
