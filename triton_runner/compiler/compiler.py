class CompiledTVMFFIKernel:
    def __init__(self, function, metadata):
        self._function = function
        self._metadata = metadata
        self._run_launcher = None

    def _get_launcher(self):
        if self._run_launcher is None:
            from triton_runner.tvm_ffi.driver import TvmFfiLauncher
            self._run_launcher = TvmFfiLauncher(self._metadata, self._function)
        return self._run_launcher

    def run(self, gridX, gridY, gridZ, launch_enter_hook, launch_exit_hook, *args):
        self._get_launcher().launch(gridX, gridY, gridZ, *args)

    def __getitem__(self, grid):
        launcher = self._get_launcher()

        def runner(*args, stream=None):
            launcher.launch(grid[0], grid[1], grid[2], *args)

        return runner
