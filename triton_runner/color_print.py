import termcolor
import os

def blue_print(text):
    print(termcolor.colored(text, "blue"), flush=True)

def yellow_print(text):
    print(termcolor.colored(text, "yellow"), flush=True)


def get_project_name():
    return "[Triton Runner]"

def warning_debug_mode_ssa_and_op(ssa, op, loc, size, encoding):
    encoding = f" with encoding={encoding[2:]}" if encoding != "" else ""
    blue_print(f"{get_project_name()} In debug mode, ssa={ssa}, op={op}, loc={loc}, size={size}{encoding}")

def warning_size_not_supported(ssa, op, loc, size):
    yellow_print(f"{get_project_name()} Warning: size={size} is not supported. And ssa={ssa}, op={op}, loc={loc}")

def print_triton_cache_dir(metadata_path, always_compile=True):
    always_compile_text = "with always_compile " if always_compile else ""
    blue_print(f"{get_project_name()} Triton cache {always_compile_text}saved at {os.path.dirname(metadata_path)}")
