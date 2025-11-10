from ..jit import jit
import triton.language as tl
from triton._C.libtriton import ir
from triton.language._utils import validate_block_shape
from triton.language.core import builtin, _unwrap_shape, cast
from triton.language.core import semantic as _semantic

def scalar_constant(value, dtype: tl.dtype, _builder):
    # scalar
    if dtype is None:
        raise ValueError("dtype must be specified when value is not a tensor")
    if value == 0:
        value = _builder.get_null_value(dtype.to_ir(_builder))
    else:
        get_value_fn = getattr(_builder, f"get_{dtype.name}")
        value = get_value_fn(value)
    return tl.tensor(value, dtype)

def make_scalar(value, dtype: tl.dtype, _builder):
    if isinstance(value, tl.tensor):
        assert value.numel.value == 1, "only accepts size-1 tensor"
        return value.to(dtype, _builder=_builder)
    return scalar_constant(value, dtype, _builder)

@builtin
def dump(val: tl.tensor, offset=0, dump_grid=None, _builder=None):
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
    pid_0 = _semantic.program_id(0, _builder)
    pid_1 = _semantic.program_id(1, _builder)
    pid_2 = _semantic.program_id(2, _builder)
    dump_pid_0_val = make_scalar(dump_pid_0, tl.int32, _builder)
    dump_pid_1_val = make_scalar(dump_pid_1, tl.int32, _builder)
    dump_pid_2_val = make_scalar(dump_pid_2, tl.int32, _builder)
    pid_0_eq = _semantic.equal(pid_0, dump_pid_0_val, _builder)
    pid_1_eq = _semantic.equal(pid_1, dump_pid_1_val, _builder)
    pid_2_eq = _semantic.equal(pid_2, dump_pid_2_val, _builder)
    pid_0_pid_1_eq = _semantic.and_(pid_0_eq, pid_1_eq, _builder)
    pid_0_pid_1_pid2_eq = _semantic.and_(pid_0_pid_1_eq, pid_2_eq, _builder)
    if_op = _builder.create_if_op([], pid_0_pid_1_pid2_eq.handle, False)
    ip, last_loc = _builder.get_insertion_point(), _builder.get_loc()
    then_block = _builder.create_block()
    _builder.set_insertion_point_to_start(then_block)
    scalar_offset = make_scalar(offset, tl.int32, _builder)
    const_zero = make_scalar(0, tl.int32, _builder)
    val = val.to(tl.float32, _builder=_builder)
    dump_val = _semantic.add(val, 0, False, _builder)
    dump_val.handle.set_attr("tt.dump", ir.make_attr([1], dump_val.handle.get_context()))
    offset_val = _semantic.add(scalar_offset, const_zero, False, _builder)
    then_block.merge_block_before(if_op.get_then_block())
    _builder.restore_insertion_point(ip)
    _builder.set_loc(last_loc)


@builtin
def dump_boundary(val: tl.tensor, offset=0, _builder=None):
    shape = val.shape
    ndim = len(shape)
    if ndim > 2:
        raise ValueError(f"Expected 1 <= ndim <= 2 but got {ndim} dimensions, you can use reshape")
    pid_0 = _semantic.program_id(0, _builder)
    pid_1 = _semantic.program_id(1, _builder)
    pid_2 = _semantic.program_id(2, _builder)
    grid_0 = _semantic.num_programs(0, _builder)
    grid_1 = _semantic.num_programs(1, _builder)
    grid_2 = _semantic.num_programs(2, _builder)
    minus_one = make_scalar(-1, tl.int32, _builder)
    dump_pid_0_val = _semantic.add(grid_0, minus_one, False, _builder)
    dump_pid_1_val = _semantic.add(grid_1, minus_one, False, _builder)
    dump_pid_2_val = _semantic.add(grid_2, minus_one, False, _builder)
    pid_0_eq = _semantic.equal(pid_0, dump_pid_0_val, _builder)
    pid_1_eq = _semantic.equal(pid_1, dump_pid_1_val, _builder)
    pid_2_eq = _semantic.equal(pid_2, dump_pid_2_val, _builder)
    pid_0_pid_1_eq = _semantic.and_(pid_0_eq, pid_1_eq, _builder)
    pid_0_pid_1_pid2_eq = _semantic.and_(pid_0_pid_1_eq, pid_2_eq, _builder)
    if_op = _builder.create_if_op([], pid_0_pid_1_pid2_eq.handle, False)
    ip, last_loc = _builder.get_insertion_point(), _builder.get_loc()
    then_block = _builder.create_block()
    _builder.set_insertion_point_to_start(then_block)
    scalar_offset = make_scalar(offset, tl.int32, _builder)
    const_zero = make_scalar(0, tl.int32, _builder)
    val = val.to(tl.float32, _builder=_builder)
    dump_val = _semantic.add(val, 0, False, _builder)
    dump_val.handle.set_attr("tt.dump", ir.make_attr([1], dump_val.handle.get_context()))
    offset_val = _semantic.add(scalar_offset, const_zero, False, _builder)
    then_block.merge_block_before(if_op.get_then_block())
    _builder.restore_insertion_point(ip)
    _builder.set_loc(last_loc)


@builtin
def dump_grids(val: tl.tensor, offset=0, _builder=None):
    shape = val.shape
    ndim = len(shape)
    if ndim > 2:
        raise ValueError(f"Expected 1 <= ndim <= 2 but got {ndim} dimensions, you can use reshape")
    pid_0 = _semantic.program_id(0, _builder)
    pid_1 = _semantic.program_id(1, _builder)
    pid_2 = _semantic.program_id(2, _builder)
    grid_0 = _semantic.num_programs(0, _builder)
    grid_1 = _semantic.num_programs(1, _builder)
    grid_2 = _semantic.num_programs(2, _builder)
    pid_1d = _semantic.mul(pid_0, grid_1, False, _builder)
    pid_1d = _semantic.add(pid_1d, pid_1, False, _builder)
    pid_1d = _semantic.mul(pid_1d, grid_2, False, _builder)
    pid_1d = _semantic.add(pid_1d, pid_2, False, _builder)
    numel = validate_block_shape(tuple(_unwrap_shape(shape)))
    numel_val = make_scalar(numel, tl.int32, _builder)
    grid_offset = _semantic.mul(pid_1d, numel_val, False, _builder)
    scalar_offset = make_scalar(offset, tl.int32, _builder)
    scalar_offset = _semantic.add(grid_offset, scalar_offset, False, _builder)
    const_zero = make_scalar(0, tl.int32, _builder)
    val = val.to(tl.float32, _builder=_builder)
    dump_val = _semantic.add(val, 0, False, _builder)
    dump_val.handle.set_attr("tt.dump", ir.make_attr([1], dump_val.handle.get_context()))
    offset_val = _semantic.add(scalar_offset, const_zero, False, _builder)
