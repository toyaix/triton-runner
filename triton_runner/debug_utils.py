

from .color_print import warning_debug_mode_ssa_and_op, warning_size_not_supported

def get_injected_ir_begin(original_line, indent, loc):
    return f"""{original_line}

{indent}// triton_runner debug start
{indent}%debug_pid_x            = tt.get_program_id x : i32 {loc}
{indent}%debug_pid_y            = tt.get_program_id y : i32 {loc}
{indent}%debug_pid_z            = tt.get_program_id z : i32 {loc}
{indent}%debug_c0_i32           = arith.constant 0 : i32 {loc}
{indent}%debug_cmpi_eq_x        = arith.cmpi eq, %debug_pid_x, %debug_c0_i32 : i32 {loc}
{indent}%debug_cmpi_eq_y        = arith.cmpi eq, %debug_pid_y, %debug_c0_i32 : i32 {loc}
{indent}%debug_cmpi_eq_z        = arith.cmpi eq, %debug_pid_z, %debug_c0_i32 : i32 {loc}
{indent}%debug_cmpi_eq_x_y      = arith.andi %debug_cmpi_eq_x, %debug_cmpi_eq_y : i1 {loc}
{indent}%debug_cmpi_eq_x_y_z    = arith.andi %debug_cmpi_eq_x_y, %debug_cmpi_eq_z : i1 {loc}
{indent}scf.if %debug_cmpi_eq_x_y_z {{"""

def get_injected_ir_end(indent):
    return f"""{indent}}}
{indent}// triton_runner debug end"""


def get_1d_injected_ir(ssa_value, original_line, indent, size, encoding, loc):
    return f"""{get_injected_ir_begin(original_line, indent, loc)}
{indent}  %debug_range          = tt.make_range {{end = {size} : i32, start = 0 : i32}} : tensor<{size}xi32{encoding}> {loc}
{indent}  %debug_splat          = tt.splat %debug_tensor : !tt.ptr<f32> -> tensor<{size}x!tt.ptr<f32>{encoding}> {loc}
{indent}  %debug_ptr            = tt.addptr %debug_splat, %debug_range : tensor<{size}x!tt.ptr<f32>{encoding}>, tensor<{size}xi32{encoding}> {loc}
{indent}  tt.store %debug_ptr, {ssa_value} : tensor<{size}x!tt.ptr<f32>{encoding}> {loc}
{get_injected_ir_end(indent)}
"""

def get_2d_injected_ir_without_encoding(ssa_value, original_line, indent, size, loc):
    size_0, size_1 = size.split('x')
    return f"""{get_injected_ir_begin(original_line, indent, loc)}
{indent}  %debug_range_1        = tt.make_range {{end = {size_1} : i32, start = 0 : i32}} : tensor<{size_1}xi32> {loc}
{indent}  %debug_expand_1       = tt.expand_dims %debug_range_1 {{axis = 0 : i32}} : tensor<{size_1}xi32> -> tensor<1x{size_1}xi32> {loc}
{indent}  %debug_broadcast_1    = tt.broadcast %debug_expand_1 : tensor<1x{size_1}xi32> -> tensor<{size}xi32> {loc}
{indent}  %debug_range_0        = tt.make_range {{end = {size_0} : i32, start = 0 : i32}} : tensor<{size_0}xi32> {loc}
{indent}  %debug_expand_0       = tt.expand_dims %debug_range_0 {{axis = 1 : i32}} : tensor<{size_0}xi32> -> tensor<{size_0}x1xi32> {loc}
{indent}  %debug_size_1_i32     = arith.constant {size_1} : i32 {loc}
{indent}  %debug_size_0_splat   = tt.splat %debug_size_1_i32 : i32 -> tensor<{size_0}x1xi32> {loc}
{indent}  %debug_size_0_off     = arith.muli %debug_expand_0, %debug_size_0_splat : tensor<{size_0}x1xi32> {loc}
{indent}  %debug_broadcast_0    = tt.broadcast %debug_size_0_off : tensor<{size_0}x1xi32> -> tensor<{size}xi32> {loc}
{indent}  %debug_range          = arith.addi %debug_broadcast_0, %debug_broadcast_1 : tensor<{size}xi32> {loc}
{indent}  %debug_splat          = tt.splat %debug_tensor : !tt.ptr<f32> -> tensor<{size}x!tt.ptr<f32>> {loc}
{indent}  %debug_ptr            = tt.addptr %debug_splat, %debug_range : tensor<{size}x!tt.ptr<f32>>, tensor<{size}xi32> {loc}
{indent}  tt.store %debug_ptr, {ssa_value} : tensor<{size}x!tt.ptr<f32>> {loc}
{get_injected_ir_end(indent)}
"""

def get_2d_injected_ir_with_encoding(ssa_value, original_line, indent, size, encoding, loc):
    size_0, size_1 = size.split('x')
    encoding = encoding[2:]
    return f"""{get_injected_ir_begin(original_line, indent, loc)}
{indent}  %debug_range_1        = tt.make_range {{end = {size_1} : i32, start = 0 : i32}} : tensor<{size_1}xi32, #ttg.slice<{{dim = 0, parent = {encoding}}}>> {loc}
{indent}  %debug_expand_1       = tt.expand_dims %debug_range_1 {{axis = 0 : i32}} : tensor<{size_1}xi32, #ttg.slice<{{dim = 0, parent = {encoding}}}>> -> tensor<1x{size_1}xi32, {encoding}> {loc}
{indent}  %debug_broadcast_1    = tt.broadcast %debug_expand_1 : tensor<1x{size_1}xi32, {encoding}> -> tensor<{size}xi32, {encoding}> {loc}
{indent}  %debug_range_0        = tt.make_range {{end = {size_0} : i32, start = 0 : i32}} : tensor<{size_0}xi32, #ttg.slice<{{dim = 1, parent = {encoding}}}>> {loc}
{indent}  %debug_expand_0       = tt.expand_dims %debug_range_0 {{axis = 1 : i32}} : tensor<{size_0}xi32, #ttg.slice<{{dim = 1, parent = {encoding}}}>> -> tensor<{size_0}x1xi32, {encoding}> {loc}
{indent}  %debug_size_1_i32     = arith.constant {size_1} : i32 {loc}
{indent}  %debug_size_0_splat   = tt.splat %debug_size_1_i32 : i32 -> tensor<{size_0}x1xi32, {encoding}> {loc}
{indent}  %debug_size_0_off     = arith.muli %debug_expand_0, %debug_size_0_splat : tensor<{size_0}x1xi32, {encoding}> {loc}
{indent}  %debug_broadcast_0    = tt.broadcast %debug_size_0_off : tensor<{size_0}x1xi32, {encoding}> -> tensor<{size}xi32, {encoding}> {loc}
{indent}  %debug_range          = arith.addi %debug_broadcast_0, %debug_broadcast_1 : tensor<{size}xi32, {encoding}> {loc}
{indent}  %debug_splat          = tt.splat %debug_tensor : !tt.ptr<f32> -> tensor<{size}x!tt.ptr<f32>, {encoding}> {loc}
{indent}  %debug_ptr            = tt.addptr %debug_splat, %debug_range : tensor<{size}x!tt.ptr<f32>, {encoding}>, tensor<{size}xi32, {encoding}> {loc}
{indent}  tt.store %debug_ptr, {ssa_value} : tensor<{size}x!tt.ptr<f32>, {encoding}> {loc}
{get_injected_ir_end(indent)}
"""


def get_injected_ir(ssa_value, op, original_line, indent, size, encoding, loc):
    loc = f"loc({loc})"
    encoding = f", {encoding}" if encoding else ""
    if size.count("x") == 0:
        warning_debug_mode_ssa_and_op(ssa_value, op, loc, size, encoding)
        return get_1d_injected_ir(ssa_value, original_line, indent, size, encoding, loc)
    elif size.count("x") == 1:
        warning_debug_mode_ssa_and_op(ssa_value, op, loc, size, encoding)
        if encoding:
            return get_2d_injected_ir_with_encoding(ssa_value, original_line, indent, size, encoding, loc)
        else:
            return get_2d_injected_ir_without_encoding(ssa_value, original_line, indent, size, loc)
    else:
        warning_size_not_supported(ssa_value, op, loc, size)
        return original_line
