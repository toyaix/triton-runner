
from .console import warning_dump_mode_ssa_and_op

def get_injected_ir_begin(original_line, indent, loc, python_dump, dump_grid):
    if isinstance(dump_grid, int):
        dump_grid = [dump_grid]
    dump_grid = list(dump_grid) + [0] * (3 - len(dump_grid))
    if_begin = "" if python_dump else f"""
{indent}%runner_dump_pid_x            = tt.get_program_id x : i32 {loc}
{indent}%runner_dump_pid_y            = tt.get_program_id y : i32 {loc}
{indent}%runner_dump_pid_z            = tt.get_program_id z : i32 {loc}
{indent}%runner_dump_pid0_i32         = arith.constant {dump_grid[0]} : i32 {loc}
{indent}%runner_dump_pid1_i32         = arith.constant {dump_grid[1]} : i32 {loc}
{indent}%runner_dump_pid2_i32         = arith.constant {dump_grid[2]} : i32 {loc}
{indent}%runner_dump_cmpi_eq_x        = arith.cmpi eq, %runner_dump_pid_x, %runner_dump_pid0_i32 : i32 {loc}
{indent}%runner_dump_cmpi_eq_y        = arith.cmpi eq, %runner_dump_pid_y, %runner_dump_pid1_i32 : i32 {loc}
{indent}%runner_dump_cmpi_eq_z        = arith.cmpi eq, %runner_dump_pid_z, %runner_dump_pid2_i32 : i32 {loc}
{indent}%runner_dump_cmpi_eq_x_y      = arith.andi %runner_dump_cmpi_eq_x, %runner_dump_cmpi_eq_y : i1 {loc}
{indent}%runner_dump_cmpi_eq_x_y_z    = arith.andi %runner_dump_cmpi_eq_x_y, %runner_dump_cmpi_eq_z : i1 {loc}
{indent}scf.if %runner_dump_cmpi_eq_x_y_z {{"""
    return f"""{original_line}\n
{indent}// triton_runner dump start{if_begin}"""

def get_injected_ir_end(indent, python_dump=False):
    if_end = "" if python_dump else f"{indent}}}\n"
    return f"""{if_end}{indent}// triton_runner dump end"""


def get_1d_injected_ir(ssa_value, ir_begin, indent, size, encoding, loc, python_dump, offset_val, replace_id):
    ir_indent = indent if python_dump else f"{indent}  "
    off_ir = f"{offset_val}" if python_dump else f"arith.constant 0 : i32 {loc}"
    id_str = f"_{replace_id}" if python_dump else f""
    return f"""{ir_begin}
{ir_indent}%runner_dump_range{id_str}          = tt.make_range {{end = {size} : i32, start = 0 : i32}} : tensor<{size}xi32{encoding}> {loc}
{ir_indent}%runner_dump_off_val{id_str}        = {off_ir}
{ir_indent}%runner_dump_with_offset{id_str}    = tt.addptr %runner_dump_tensor, %runner_dump_off_val{id_str} : !tt.ptr<f32>, i32 {loc}
{ir_indent}%runner_dump_splat{id_str}          = tt.splat %runner_dump_with_offset{id_str} : !tt.ptr<f32> -> tensor<{size}x!tt.ptr<f32>{encoding}> {loc}
{ir_indent}%runner_dump_ptr{id_str}            = tt.addptr %runner_dump_splat{id_str}, %runner_dump_range{id_str} : tensor<{size}x!tt.ptr<f32>{encoding}>, tensor<{size}xi32{encoding}> {loc}
{ir_indent}tt.store %runner_dump_ptr{id_str}, {ssa_value} : tensor<{size}x!tt.ptr<f32>{encoding}> {loc}
{get_injected_ir_end(indent, python_dump)}
"""

def get_nd_injected_ir_without_encoding(ssa_value, ir_begin, indent, size, elem_ty, loc, python_dump, offset_val, replace_id):
    ir_indent = indent if python_dump else f"{indent}  "
    id_str = f"_{replace_id}" if python_dump else f""
    dims = size.split('x')
    flat_size = 1
    for d in dims:
        flat_size *= int(d)
    flat_size = str(flat_size)
    ir_begin = f"""{ir_begin}
{ir_indent}%runner_dump_reshape{id_str}        = tt.reshape {ssa_value} : tensor<{size}x{elem_ty}> -> tensor<{flat_size}x{elem_ty}> {loc}"""
    return get_1d_injected_ir(f"%runner_dump_reshape{id_str}", ir_begin, indent, flat_size, "", loc, python_dump, offset_val, replace_id)

def _get_slice_encoding(encoding, removed_dims):
    slice_encoding = encoding
    for dim in sorted(removed_dims, reverse=True):
        slice_encoding = f"#ttg.slice<{{dim = {dim}, parent = {slice_encoding}}}>"
    return slice_encoding

def _shape_str(shape_vals):
    return "x".join(shape_vals)

def get_nd_injected_ir_with_encoding(ssa_value, ir_begin, indent, size, encoding, loc):
    dims = size.split('x')
    rank = len(dims)
    encoding = encoding[2:]
    full_shape = _shape_str(dims)
    lines = [ir_begin]
    contribs = []

    for dim_idx, dim_size in enumerate(dims):
        other_dims = [idx for idx in range(rank) if idx != dim_idx]
        current_encoding = _get_slice_encoding(encoding, other_dims)
        current_name = f"%runner_dump_range_{dim_idx}"
        current_axes = [dim_idx]
        current_shape = [dim_size]
        lines.append(
            f"{indent}  {current_name}        = tt.make_range {{end = {dim_size} : i32, start = 0 : i32}} : "
            f"tensor<{dim_size}xi32, {current_encoding}> {loc}"
        )

        expanded_dims = []
        for axis in sorted(other_dims):
            next_axes = sorted(current_axes + [axis])
            next_shape = current_shape.copy()
            next_shape.insert(next_axes.index(axis), "1")
            next_encoding = _get_slice_encoding(encoding, [idx for idx in other_dims if idx not in expanded_dims + [axis]])
            next_name = f"%runner_dump_expand_{dim_idx}_{axis}"
            lines.append(
                f"{indent}  {next_name}       = tt.expand_dims {current_name} {{axis = {axis} : i32}} : "
                f"tensor<{_shape_str(current_shape)}xi32, {current_encoding}> -> "
                f"tensor<{_shape_str(next_shape)}xi32, {next_encoding}> {loc}"
            )
            current_name = next_name
            current_axes = next_axes
            current_shape = next_shape
            current_encoding = next_encoding
            expanded_dims.append(axis)

        stride = 1
        for dim in dims[dim_idx + 1:]:
            stride *= int(dim)
        current_shape_str = _shape_str(current_shape)
        if stride != 1:
            stride_i32 = f"%runner_dump_stride_{dim_idx}_i32"
            stride_splat = f"%runner_dump_stride_{dim_idx}_splat"
            scaled_name = f"%runner_dump_offset_{dim_idx}"
            lines.append(f"{indent}  {stride_i32}     = arith.constant {stride} : i32 {loc}")
            lines.append(
                f"{indent}  {stride_splat}   = tt.splat {stride_i32} : i32 -> tensor<{current_shape_str}xi32, {encoding}> {loc}"
            )
            lines.append(
                f"{indent}  {scaled_name}  = arith.muli {current_name}, {stride_splat} : tensor<{current_shape_str}xi32, {encoding}> {loc}"
            )
            current_name = scaled_name
        contribs.append((current_name, current_shape_str))

    offset_name, offset_shape = contribs[0]
    for idx, (contrib_name, contrib_shape) in enumerate(contribs[1:], start=1):
        target_shape = _shape_str([dim if axis <= idx else "1" for axis, dim in enumerate(dims)])
        if offset_shape != target_shape:
            next_name = f"%runner_dump_broadcast_acc_{idx}"
            lines.append(
                f"{indent}  {next_name} = tt.broadcast {offset_name} : "
                f"tensor<{offset_shape}xi32, {encoding}> -> tensor<{target_shape}xi32, {encoding}> {loc}"
            )
            offset_name = next_name
            offset_shape = target_shape
        if contrib_shape != target_shape:
            next_name = f"%runner_dump_broadcast_rhs_{idx}"
            lines.append(
                f"{indent}  {next_name} = tt.broadcast {contrib_name} : "
                f"tensor<{contrib_shape}xi32, {encoding}> -> tensor<{target_shape}xi32, {encoding}> {loc}"
            )
            contrib_name = next_name
            contrib_shape = target_shape
        next_name = f"%runner_dump_range_add_{idx}"
        lines.append(
            f"{indent}  {next_name} = arith.addi {offset_name}, {contrib_name} : tensor<{target_shape}xi32, {encoding}> {loc}"
        )
        offset_name = next_name
        offset_shape = target_shape

    if offset_shape != full_shape:
        next_name = "%runner_dump_broadcast_final"
        lines.append(
            f"{indent}  {next_name} = tt.broadcast {offset_name} : "
            f"tensor<{offset_shape}xi32, {encoding}> -> tensor<{full_shape}xi32, {encoding}> {loc}"
        )
        offset_name = next_name

    lines.extend([
        f"{indent}  %runner_dump_splat          = tt.splat %runner_dump_tensor : !tt.ptr<f32> -> tensor<{full_shape}x!tt.ptr<f32>, {encoding}> {loc}",
        f"{indent}  %runner_dump_ptr            = tt.addptr %runner_dump_splat, {offset_name} : tensor<{full_shape}x!tt.ptr<f32>, {encoding}>, tensor<{full_shape}xi32, {encoding}> {loc}",
        f"{indent}  tt.store %runner_dump_ptr, {ssa_value} : tensor<{full_shape}x!tt.ptr<f32>, {encoding}> {loc}",
        get_injected_ir_end(indent),
    ])
    return "\n".join(lines) + "\n"


def get_injected_ir(ssa_value, op, original_line, indent, size, elem_ty, encoding, loc, python_dump=False, offset_val="", replace_id=0, dump_grid=(0,0,0)):
    loc = f"loc({loc})"
    encoding = f", {encoding}" if encoding else ""
    ir_begin = get_injected_ir_begin(original_line, indent, loc, python_dump, dump_grid)
    warning_dump_mode_ssa_and_op(ssa_value, op, loc, size, encoding)
    if size.count("x") == 0:
        return get_1d_injected_ir(ssa_value, ir_begin, indent, size, encoding, loc, python_dump, offset_val, replace_id)
    elif encoding:
            return get_nd_injected_ir_with_encoding(ssa_value, ir_begin, indent, size, encoding, loc)
    else:
        return get_nd_injected_ir_without_encoding(ssa_value, ir_begin, indent, size, elem_ty, loc, python_dump, offset_val, replace_id)
