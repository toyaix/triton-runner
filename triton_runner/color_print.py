import termcolor
import os


def warning_debug_mode_ssa_and_op(ssa, op, loc, size):
    blue_print(f"[triton-runner] In debug mode, ssa={ssa}, op={op}, loc={loc}, size={size}")

def blue_print(text):
    print(termcolor.colored(text, "blue"), flush=True)

def yellow_print(text):
    print(termcolor.colored(text, "yellow"), flush=True)

def warning_size_not_supported(ssa, op, loc, size):
    yellow_print(f"[triton-runner] Warning: size={size} is not supported. And ssa={ssa}, op={op}, loc={loc}")

def print_triton_cache_dir(metadata_path, always_compile=True):
    always_compile_text = "with always_compile " if always_compile else ""
    blue_print(f"[triton-runner] Triton cache {always_compile_text}saved at {os.path.dirname(metadata_path)}")
