import functools

class KernelLauncher:
    def __init__(
        self,
        mod=None,
        grid_x=1,
        grid_y=1,
        grid_z=1,
        _stream=None,
        function=None,
        launch_cooperative_grid=False,
        global_scratch=None,
        packed_metadata=None,
        launch_metadata=None,
        launch_enter_hook=None,
        launch_exit_hook=None,
        bound_args=None,
    ):
        self.mod = mod
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.grid_z = grid_z
        self._stream = _stream
        self.function = function
        self.launch_cooperative_grid = launch_cooperative_grid
        self.global_scratch = global_scratch
        self.packed_metadata = packed_metadata
        self.launch_metadata = launch_metadata
        self.launch_enter_hook = launch_enter_hook
        self.launch_exit_hook = launch_exit_hook
        self.bound_args = bound_args

    def __repr__(self):
        return (
            f"KernelLauncher(mod={self.mod}, grid=({self.grid_x}, {self.grid_y}, {self.grid_z}), "
            f"_stream={self._stream}, function={self.function}, "
            f"cooperative={self.launch_cooperative_grid}, global_scratch={self.global_scratch}, "
            f"packed_metadata={self.packed_metadata}, launch_metadata={self.launch_metadata}, "
            f"enter_hook={self.launch_enter_hook}, exit_hook={self.launch_exit_hook}, "
            f"bound_args={self.bound_args})"
        )

    @functools.lru_cache()
    def run(self):
        self.mod.launch(
            self.grid_x,
            self.grid_y,
            self.grid_z,
            self._stream,
            self.function,
            self.launch_cooperative_grid,
            self.global_scratch,
            self.packed_metadata,
            self.launch_metadata,
            self.launch_enter_hook,
            self.launch_exit_hook,
            *self.bound_args
        )
