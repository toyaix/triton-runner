import triton
from ..version_utils import is_triton_geq_v3_4

if is_triton_geq_v3_4:
    from .dump import dump, dump_boundary, dump_grids
else:
    from .dump_before_3_4_0 import dump, dump_boundary, dump_grids
