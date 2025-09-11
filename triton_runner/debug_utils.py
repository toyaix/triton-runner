

from .color_print import warning_debug_mode_ssa_and_op, warning_size_not_supported


def get_1d_injected_ir(ssa_value, original_line, indent, size, loc):
    return f"""{original_line}
{indent}%debuger_range          = tt.make_range {{end = {size} : i32, start = 0 : i32}} : tensor<{size}xi32> loc({loc})
{indent}%debuger_splat          = tt.splat %debug_tensor : !tt.ptr<f32> -> tensor<{size}x!tt.ptr<f32>> loc({loc})
{indent}%debuger_ptr            = tt.addptr %debuger_splat, %debuger_range : tensor<{size}x!tt.ptr<f32>>, tensor<{size}xi32> loc({loc})
{indent}tt.store %debuger_ptr, {ssa_value} : tensor<{size}x!tt.ptr<f32>> loc({loc})"""


def get_2d_injected_ir(ssa_value, original_line, indent, size, loc):
    size_0, size_1 = size.split('x')
    print(size_0, size_1)
    return f"""{original_line}
{indent}%debuger_range_0        = tt.make_range {{end = {size_0} : i32, start = 0 : i32}} : tensor<{size_0}xi32> loc({loc})
{indent}%debug_expand_0         = tt.expand_dims %debuger_range_0 {{axis = 0 : i32}} : tensor<{size_0}xi32> -> tensor<1x{size_0}xi32> loc({loc})
{indent}%debug_broadcast_0      = tt.broadcast %debug_expand_0 : tensor<1x{size_0}xi32> -> tensor<{size}xi32> loc({loc})
{indent}%debuger_range_1        = tt.make_range {{end = {size_1} : i32, start = 0 : i32}} : tensor<{size_1}xi32> loc({loc})
{indent}%debug_expand_1         = tt.expand_dims %debuger_range_1 {{axis = 1 : i32}} : tensor<{size_1}xi32> -> tensor<{size_1}x1xi32> loc({loc})
{indent}%debuger_size_0_i32     = arith.constant {size_0} : i32 loc({loc})
{indent}%debuger_size_0_splat   = tt.splat %debuger_size_0_i32 : i32 -> tensor<{size_0}x1xi32> loc({loc})
{indent}%debug_28               = arith.muli %debug_expand_1, %debuger_size_0_splat : tensor<32x1xi32> loc({loc})
{indent}%debug_broadcast_1      = tt.broadcast %debug_28 : tensor<{size_1}x1xi32> -> tensor<{size}xi32> loc({loc})
{indent}%debuger_range          = arith.addi %debug_broadcast_0, %debug_broadcast_1 : tensor<{size}xi32> loc({loc})
{indent}%debuger_splat          = tt.splat %debug_tensor : !tt.ptr<f32> -> tensor<{size}x!tt.ptr<f32>> loc({loc})
{indent}%debuger_ptr            = tt.addptr %debuger_splat, %debuger_range : tensor<{size}x!tt.ptr<f32>>, tensor<{size}xi32> loc({loc})
{indent}tt.store %debuger_ptr, {ssa_value} : tensor<{size}x!tt.ptr<f32>> loc({loc})"""



def get_injected_ir(ssa_value, op, original_line, indent, size, loc):
    if size.count("x") == 0:
        warning_debug_mode_ssa_and_op(ssa_value, op, loc, size)
        return get_1d_injected_ir(ssa_value, original_line, indent, size, loc)
    elif size.count("x") == 1:
        warning_debug_mode_ssa_and_op(ssa_value, op, loc, size)
        return get_2d_injected_ir(ssa_value, original_line, indent, size, loc)
    else:
        warning_size_not_supported(ssa_value, op, loc, size)
        return original_line
