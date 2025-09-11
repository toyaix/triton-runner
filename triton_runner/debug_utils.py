

from .color_print import warning_debug_mode_ssa_and_op, warning_size_not_supported

def get_1d_injected_ir(ssa_value, original_line, indent, size, loc):
    return f"""{original_line}
{indent}%debuger_range          = tt.make_range {{end = {size} : i32, start = 0 : i32}} : tensor<{size}xi32> loc({loc})
{indent}%debuger_splat          = tt.splat %debug_tensor : !tt.ptr<f32> -> tensor<{size}x!tt.ptr<f32>> loc({loc})
{indent}%debuger_ptr            = tt.addptr %debuger_splat, %debuger_range : tensor<{size}x!tt.ptr<f32>>, tensor<{size}xi32> loc({loc})
{indent}tt.store %debuger_ptr, {ssa_value} : tensor<{size}x!tt.ptr<f32>> loc({loc})"""


def get_injected_ir(ssa_value, op, original_line, indent, size, loc):
    if size.count("x") == 0:
        warning_debug_mode_ssa_and_op(ssa_value, op, loc, size)
        return get_1d_injected_ir(ssa_value, original_line, indent, size, loc)
    else:
        warning_size_not_supported(ssa_value, op, loc, size)
        return original_line
