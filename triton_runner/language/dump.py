from ..jit import jit
import triton.language as tl
from triton._C.libtriton import ir
from typing import List
from triton.language.core import builtin

@builtin
def dump(val: tl.tensor, offset=0, pid_0=0, pid_1=0, pid_2=0, _semantic=None):
    query_pid_0 = _semantic.program_id(0)
    query_pid_1 = _semantic.program_id(1)
    query_pid_2 = _semantic.program_id(2)
    pid_0_val = _semantic.make_scalar(pid_0, tl.int32)
    pid_1_val = _semantic.make_scalar(pid_1, tl.int32)
    pid_2_val = _semantic.make_scalar(pid_2, tl.int32)
    pid_0_eq = _semantic.equal(query_pid_0, pid_0_val)
    pid_1_eq = _semantic.equal(query_pid_1, pid_1_val)
    pid_2_eq = _semantic.equal(query_pid_2, pid_2_val)
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
def dump_grids(val: tl.tensor, offset, _semantic=None):
    scalar_offset = _semantic.make_scalar(offset, tl.int32)
    const_zero = _semantic.make_scalar(0, tl.int32)
    val = val.to(tl.float32, _semantic=_semantic)
    dump_val = _semantic.add(val, 0, False)
    dump_val.handle.set_attr("tt.dump", ir.make_attr([1], dump_val.handle.get_context()))
    offset_val = _semantic.add(scalar_offset, const_zero, False)
