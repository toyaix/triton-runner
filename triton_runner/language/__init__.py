import triton

if triton.__version__ in ["3.4.0", "3.5.0"]:
    from .dump import dump, dump_boundary, dump_grids
else:
    from .dump_before_3_4_0 import dump, dump_boundary, dump_grids
