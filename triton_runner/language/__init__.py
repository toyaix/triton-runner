from ..compat.version import is_triton_geq_v3_4

if is_triton_geq_v3_4:
    from ._impl.v3_4_plus import dump, dump_boundary, dump_grids
else:
    from ._impl.pre_v3_4 import dump, dump_boundary, dump_grids

__all__ = ["dump", "dump_boundary", "dump_grids"]
