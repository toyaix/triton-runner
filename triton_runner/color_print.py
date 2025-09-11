import termcolor


def warning_debug_mode_grid():
    blue_print(f"In debug mode, grid is changed to (1, 1, 1)")

def warning_debug_mode_ssa_and_op(ssa, op, loc, size):
    blue_print(f"In debug mode, ssa={ssa}, op={op}, loc={loc}, size={size}")

def blue_print(text):
    print(termcolor.colored(text, "blue"), flush=True)

def yellow_print(text):
    print(termcolor.colored(text, "yellow"), flush=True)

def warning_size_not_supported(ssa, op, loc, size):
    yellow_print(f"Warning: size={size} is not supported. And ssa={ssa}, op={op}, loc={loc}")
