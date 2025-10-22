

from .color_print import warning_dump_mode_ssa_and_op, warning_size_not_supported

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
    if_end = "" if python_dump else f"\n{indent}}}"
    return f"""{if_end}{indent}// triton_runner dump end"""


def get_1d_injected_ir(ssa_value, ir_begin, indent, size, encoding, loc, python_dump, offset_val, replace_id):
    ir_indent = indent if python_dump else f"{indent}  "
    off_ir = f"{offset_val}" if python_dump else f"arith.constant 0 : i32 {loc}"
    id_str = f"_{replace_id}" if python_dump else f""
    return f"""{ir_begin}
{ir_indent}%runner_dump_range{id_str}          = tt.make_range {{end = {size} : i32, start = 0 : i32}} : tensor<{size}xi32{encoding}> {loc}
{ir_indent}%runner_dump_off_val{id_str}        = {off_ir}
{ir_indent}%runner_dump_with_offset{id_str}    = tt.addptr %runner_dump_tensor, %runner_dump_off_val{id_str} : !tt.ptr<f32>, i32 loc(#loc1)
{ir_indent}%runner_dump_splat{id_str}          = tt.splat %runner_dump_with_offset{id_str} : !tt.ptr<f32> -> tensor<{size}x!tt.ptr<f32>{encoding}> {loc}
{ir_indent}%runner_dump_ptr{id_str}            = tt.addptr %runner_dump_splat{id_str}, %runner_dump_range{id_str} : tensor<{size}x!tt.ptr<f32>{encoding}>, tensor<{size}xi32{encoding}> {loc}
{ir_indent}tt.store %runner_dump_ptr{id_str}, {ssa_value} : tensor<{size}x!tt.ptr<f32>{encoding}> {loc}
{get_injected_ir_end(indent, python_dump)}
"""

def get_2d_injected_ir_without_encoding(ssa_value, ir_begin, indent, size, loc, python_dump, offset_val, replace_id):
    ir_indent = indent if python_dump else f"{indent}  "
    off_ir = f"{offset_val}" if python_dump else f"arith.constant 0 : i32 {loc}"
    id_str = f"_{replace_id}" if python_dump else f""
    size_0, size_1 = size.split('x')
    return f"""{ir_begin}
{ir_indent}%runner_dump_range_1{id_str}        = tt.make_range {{end = {size_1} : i32, start = 0 : i32}} : tensor<{size_1}xi32> {loc}
{ir_indent}%runner_dump_expand_1{id_str}       = tt.expand_dims %runner_dump_range_1{id_str} {{axis = 0 : i32}} : tensor<{size_1}xi32> -> tensor<1x{size_1}xi32> {loc}
{ir_indent}%runner_dump_broadcast_1{id_str}    = tt.broadcast %runner_dump_expand_1{id_str} : tensor<1x{size_1}xi32> -> tensor<{size}xi32> {loc}
{ir_indent}%runner_dump_range_0{id_str}        = tt.make_range {{end = {size_0} : i32, start = 0 : i32}} : tensor<{size_0}xi32> {loc}
{ir_indent}%runner_dump_expand_0{id_str}       = tt.expand_dims %runner_dump_range_0{id_str} {{axis = 1 : i32}} : tensor<{size_0}xi32> -> tensor<{size_0}x1xi32> {loc}
{ir_indent}%runner_dump_size_1_i32{id_str}     = arith.constant {size_1} : i32 {loc}
{ir_indent}%runner_dump_size_0_splat{id_str}   = tt.splat %runner_dump_size_1_i32{id_str} : i32 -> tensor<{size_0}x1xi32> {loc}
{ir_indent}%runner_dump_size_0_off{id_str}     = arith.muli %runner_dump_expand_0{id_str}, %runner_dump_size_0_splat{id_str} : tensor<{size_0}x1xi32> {loc}
{ir_indent}%runner_dump_broadcast_0{id_str}    = tt.broadcast %runner_dump_size_0_off{id_str} : tensor<{size_0}x1xi32> -> tensor<{size}xi32> {loc}
{ir_indent}%runner_dump_range{id_str}          = arith.addi %runner_dump_broadcast_0{id_str}, %runner_dump_broadcast_1{id_str} : tensor<{size}xi32> {loc}
{ir_indent}%runner_dump_off_val{id_str}        = {off_ir}
{ir_indent}%runner_dump_with_offset{id_str}    = tt.addptr %runner_dump_tensor, %runner_dump_off_val{id_str} : !tt.ptr<f32>, i32 loc(#loc1)
{ir_indent}%runner_dump_splat{id_str}          = tt.splat %runner_dump_with_offset{id_str} : !tt.ptr<f32> -> tensor<{size}x!tt.ptr<f32>> {loc}
{ir_indent}%runner_dump_ptr{id_str}            = tt.addptr %runner_dump_splat{id_str}, %runner_dump_range{id_str} : tensor<{size}x!tt.ptr<f32>>, tensor<{size}xi32> {loc}
{ir_indent}tt.store %runner_dump_ptr{id_str}, {ssa_value} : tensor<{size}x!tt.ptr<f32>> {loc}
{get_injected_ir_end(indent, python_dump)}
"""

def get_2d_injected_ir_with_encoding(ssa_value, ir_begin, indent, size, encoding, loc):
    size_0, size_1 = size.split('x')
    encoding = encoding[2:]
    return f"""{ir_begin}
{indent}  %runner_dump_range_1        = tt.make_range {{end = {size_1} : i32, start = 0 : i32}} : tensor<{size_1}xi32, #ttg.slice<{{dim = 0, parent = {encoding}}}>> {loc}
{indent}  %runner_dump_expand_1       = tt.expand_dims %runner_dump_range_1 {{axis = 0 : i32}} : tensor<{size_1}xi32, #ttg.slice<{{dim = 0, parent = {encoding}}}>> -> tensor<1x{size_1}xi32, {encoding}> {loc}
{indent}  %runner_dump_broadcast_1    = tt.broadcast %runner_dump_expand_1 : tensor<1x{size_1}xi32, {encoding}> -> tensor<{size}xi32, {encoding}> {loc}
{indent}  %runner_dump_range_0        = tt.make_range {{end = {size_0} : i32, start = 0 : i32}} : tensor<{size_0}xi32, #ttg.slice<{{dim = 1, parent = {encoding}}}>> {loc}
{indent}  %runner_dump_expand_0       = tt.expand_dims %runner_dump_range_0 {{axis = 1 : i32}} : tensor<{size_0}xi32, #ttg.slice<{{dim = 1, parent = {encoding}}}>> -> tensor<{size_0}x1xi32, {encoding}> {loc}
{indent}  %runner_dump_size_1_i32     = arith.constant {size_1} : i32 {loc}
{indent}  %runner_dump_size_0_splat   = tt.splat %runner_dump_size_1_i32 : i32 -> tensor<{size_0}x1xi32, {encoding}> {loc}
{indent}  %runner_dump_size_0_off     = arith.muli %runner_dump_expand_0, %runner_dump_size_0_splat : tensor<{size_0}x1xi32, {encoding}> {loc}
{indent}  %runner_dump_broadcast_0    = tt.broadcast %runner_dump_size_0_off : tensor<{size_0}x1xi32, {encoding}> -> tensor<{size}xi32, {encoding}> {loc}
{indent}  %runner_dump_range          = arith.addi %runner_dump_broadcast_0, %runner_dump_broadcast_1 : tensor<{size}xi32, {encoding}> {loc}
{indent}  %runner_dump_splat          = tt.splat %runner_dump_tensor : !tt.ptr<f32> -> tensor<{size}x!tt.ptr<f32>, {encoding}> {loc}
{indent}  %runner_dump_ptr            = tt.addptr %runner_dump_splat, %runner_dump_range : tensor<{size}x!tt.ptr<f32>, {encoding}>, tensor<{size}xi32, {encoding}> {loc}
{indent}  tt.store %runner_dump_ptr, {ssa_value} : tensor<{size}x!tt.ptr<f32>, {encoding}> {loc}
{get_injected_ir_end(indent)}
"""


def get_injected_ir(ssa_value, op, original_line, indent, size, encoding, loc, python_dump=False, offset_val="", replace_id=0, dump_grid=(0,0,0)):
    loc = f"loc({loc})"
    encoding = f", {encoding}" if encoding else ""
    ir_begin = get_injected_ir_begin(original_line, indent, loc, python_dump, dump_grid)
    warning_dump_mode_ssa_and_op(ssa_value, op, loc, size, encoding)
    if size.count("x") == 0:
        return get_1d_injected_ir(ssa_value, ir_begin, indent, size, encoding, loc, python_dump, offset_val, replace_id)
    elif size.count("x") == 1:
        if encoding:
            return get_2d_injected_ir_with_encoding(ssa_value, ir_begin, indent, size, encoding, loc)
        else:
            return get_2d_injected_ir_without_encoding(ssa_value, ir_begin, indent, size, loc, python_dump, offset_val, replace_id)
    else:
        warning_size_not_supported(ssa_value, op, loc, size)
        return original_line
