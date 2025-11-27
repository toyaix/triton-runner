from pathlib import Path
import copy

from triton import knobs
from triton.compiler.compiler import CompiledKernel, max_shared_mem, json, GPUTarget, make_backend, AsmDict, ASTSource, LazyDict
from triton.runtime.driver import driver
from triton.runtime.autotuner import OutOfResources
import functools



def _raise_error(err, *args, **kwargs):
    raise copy.deepcopy(err)


class CompiledKernel_v3_5_0(CompiledKernel):

    def __init__(self, src, metadata_group, hash):
        from collections import namedtuple
        metadata_path = next((Path(p) for c, p in metadata_group.items() if c.endswith(".json")))
        metadata = json.loads(metadata_path.read_text())
        metadata['cluster_dims'] = tuple(metadata['cluster_dims'])
        # JSON serialization dumps the target as a dict. Restore it to a GPUTarget.
        target = metadata['target']
        metadata['target'] = GPUTarget(target['backend'], target['arch'], target['warp_size'])
        KernelMetadata = namedtuple('KernelMetadata', sorted(list(metadata.keys())))
        self.metadata = KernelMetadata(**metadata)
        backend = make_backend(self.metadata.target)
        self.packed_metadata = backend.pack_metadata(self.metadata)
        self.src = src
        self.hash = hash
        self.name = self.metadata.name
        # stores the text of each level of IR that was generated during compilation
        asm_files = [Path(p) for c, p in metadata_group.items() if not c.endswith(".json")]
        binary_ext = backend.binary_ext
        self.asm = AsmDict({
            file.suffix[1:]: file.read_bytes() if file.suffix[1:] == binary_ext else file.read_text()
            for file in asm_files
        })
        self.metadata_group = metadata_group
        self.kernel = self.asm[binary_ext]
        # binaries are lazily initialized
        # because it involves doing runtime things
        # (e.g., checking amount of shared memory on current device)
        self.module = None
        self.function = None
        self._run = None

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
    @property
    def run(self):
        if self._run is None:
            self._init_handles()
        return self._run

    def launch_metadata(self, grid, stream, *args):
        if knobs.runtime.launch_enter_hook is None:
            return None
        self._init_handles()
        ret = LazyDict({"name": self.name, "function": self.function, "stream": stream})
        if not isinstance(self.src, ASTSource) or self.src.fn.launch_metadata is None:
            return ret
        arg_dict = {name: arg for name, arg in zip(self.src.fn.arg_names, args)}
        ret.add(self.src.fn.launch_metadata, (grid, self.metadata, arg_dict))
        return ret

    def __getitem__(self, grid):
        self._init_handles()

        def runner(*args, stream=None):
            if stream is None:
                device = driver.active.get_current_device()
                stream = driver.active.get_current_stream(device)
            launch_metadata = self.launch_metadata(grid, stream, *args)
            self.run(grid[0], grid[1], grid[2], stream, self.function, self.packed_metadata, launch_metadata,
                     knobs.runtime.launch_enter_hook, knobs.runtime.launch_exit_hook, *args)

        return runner
