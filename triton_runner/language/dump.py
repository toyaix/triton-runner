from ..jit import jit
import triton.language as tl
from triton._C.libtriton import ir
from triton._utils import validate_block_shape
from triton.language.core import builtin, _unwrap_shape

@builtin
def dump(val: tl.tensor, offset=0, dump_grid=None, _semantic=None):
    shape = val.shape
    ndim = len(shape)
    if ndim > 2:
        raise ValueError(f"Expected 1 <= ndim <= 2 but got {ndim} dimensions, you can use reshape")
    dump_pid_0, dump_pid_1, dump_pid_2 = 0, 0, 0
    if dump_grid:
        if isinstance(dump_grid, tl.constexpr) or isinstance(dump_grid, tl.tensor):
            dump_grid = [dump_grid]
        dump_grid = dump_grid + [0] * (3 - len(dump_grid))
        dump_pid_0, dump_pid_1, dump_pid_2 = tuple(dump_grid)
    pid_0 = _semantic.program_id(0)
    pid_1 = _semantic.program_id(1)
    pid_2 = _semantic.program_id(2)
    dump_pid_0_val = _semantic.make_scalar(dump_pid_0, tl.int32)
    dump_pid_1_val = _semantic.make_scalar(dump_pid_1, tl.int32)
    dump_pid_2_val = _semantic.make_scalar(dump_pid_2, tl.int32)
    pid_0_eq = _semantic.equal(pid_0, dump_pid_0_val)
    pid_1_eq = _semantic.equal(pid_1, dump_pid_1_val)
    pid_2_eq = _semantic.equal(pid_2, dump_pid_2_val)
    pid_0_pid_1_eq = _semantic.and_(pid_0_eq, pid_1_eq)
    pid_0_pid_1_pid2_eq = _semantic.and_(pid_0_pid_1_eq, pid_2_eq)
    if_op = _semantic.builder.create_if_op([], pid_0_pid_1_pid2_eq.handle, False)
    ip, last_loc = _semantic.builder.get_insertion_point(), _semantic.builder.get_loc()
    then_block = _semantic.builder.create_block()
    _semantic.builder.set_insertion_point_to_start(then_block)
    scalar_offset = _semantic.make_scalar(offset, tl.int32)
    const_zero = _semantic.make_scalar(0, tl.int32)
    val = val.to(tl.float32, _semantic=_semantic)
    dump_val = _semantic.add(val, 0, False)
    dump_val.handle.set_attr("tt.dump", ir.make_attr([1], dump_val.handle.get_context()))
    offset_val = _semantic.add(scalar_offset, const_zero, False)
    then_block.merge_block_before(if_op.get_then_block())
    _semantic.builder.restore_insertion_point(ip)
    _semantic.builder.set_loc(last_loc)


@builtin
def dump_boundary(val: tl.tensor, offset=0, _semantic=None):
    shape = val.shape
    ndim = len(shape)
    if ndim > 2:
        raise ValueError(f"Expected 1 <= ndim <= 2 but got {ndim} dimensions, you can use reshape")
    pid_0 = _semantic.program_id(0)
    pid_1 = _semantic.program_id(1)
    pid_2 = _semantic.program_id(2)
    grid_0 = _semantic.num_programs(0)
    grid_1 = _semantic.num_programs(1)
    grid_2 = _semantic.num_programs(2)
    minus_one = _semantic.make_scalar(-1, tl.int32)
    dump_pid_0_val = _semantic.add(grid_0, minus_one, False)
    dump_pid_1_val = _semantic.add(grid_1, minus_one, False)
    dump_pid_2_val = _semantic.add(grid_2, minus_one, False)
    pid_0_eq = _semantic.equal(pid_0, dump_pid_0_val)
    pid_1_eq = _semantic.equal(pid_1, dump_pid_1_val)
    pid_2_eq = _semantic.equal(pid_2, dump_pid_2_val)
    pid_0_pid_1_eq = _semantic.and_(pid_0_eq, pid_1_eq)
    pid_0_pid_1_pid2_eq = _semantic.and_(pid_0_pid_1_eq, pid_2_eq)
    if_op = _semantic.builder.create_if_op([], pid_0_pid_1_pid2_eq.handle, False)
    ip, last_loc = _semantic.builder.get_insertion_point(), _semantic.builder.get_loc()
    then_block = _semantic.builder.create_block()
    _semantic.builder.set_insertion_point_to_start(then_block)
    scalar_offset = _semantic.make_scalar(offset, tl.int32)
    const_zero = _semantic.make_scalar(0, tl.int32)
    val = val.to(tl.float32, _semantic=_semantic)
    dump_val = _semantic.add(val, 0, False)
    dump_val.handle.set_attr("tt.dump", ir.make_attr([1], dump_val.handle.get_context()))
    offset_val = _semantic.add(scalar_offset, const_zero, False)
    then_block.merge_block_before(if_op.get_then_block())
    _semantic.builder.restore_insertion_point(ip)
    _semantic.builder.set_loc(last_loc)


@builtin
def dump_grids(val: tl.tensor, offset=0, _semantic=None):
    shape = val.shape
    ndim = len(shape)
    if ndim > 2:
        raise ValueError(f"Expected 1 <= ndim <= 2 but got {ndim} dimensions, you can use reshape")
    pid_0 = _semantic.program_id(0)
    pid_1 = _semantic.program_id(1)
    pid_2 = _semantic.program_id(2)
    grid_0 = _semantic.num_programs(0)
    grid_1 = _semantic.num_programs(1)
    grid_2 = _semantic.num_programs(2)
    pid_1d = _semantic.mul(pid_0, grid_1, False)
    pid_1d = _semantic.add(pid_1d, pid_1, False)
    pid_1d = _semantic.mul(pid_1d, grid_2, False)
    pid_1d = _semantic.add(pid_1d, pid_2, False)
    numel = validate_block_shape(tuple(_unwrap_shape(shape)))
    numel_val = _semantic.make_scalar(numel, tl.int32)
    grid_offset = _semantic.mul(pid_1d, numel_val, False)
    scalar_offset = _semantic.make_scalar(offset, tl.int32)
    scalar_offset = _semantic.add(grid_offset, scalar_offset, False)
    const_zero = _semantic.make_scalar(0, tl.int32)
    val = val.to(tl.float32, _semantic=_semantic)
    dump_val = _semantic.add(val, 0, False)
    dump_val.handle.set_attr("tt.dump", ir.make_attr([1], dump_val.handle.get_context()))
    offset_val = _semantic.add(scalar_offset, const_zero, False)
