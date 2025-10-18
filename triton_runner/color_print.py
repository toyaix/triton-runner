import termcolor
import os

def blue_print(text):
    print(termcolor.colored(text, "blue"), flush=True)

def yellow_print(text):
    print(termcolor.colored(text, "yellow"), flush=True)


def get_project_name():
    return "[Triton Runner]"

def warning_dump_mode_ssa_and_op(ssa, op, loc, size, encoding):
    encoding = f" with encoding={encoding[2:]}" if encoding != "" else ""
    blue_print(f"{get_project_name()} In dump mode, ssa={ssa}, op={op}, loc={loc}, size={size}{encoding}")

def warning_size_not_supported(ssa, op, loc, size):
    yellow_print(f"{get_project_name()} Warning: size={size} is not supported. And ssa={ssa}, op={op}, loc={loc}")

def print_triton_cache_dir(metadata_path, cache_hit=False):
    if os.environ.get("RUNNER_PROD", "0") != "1":
        always_compile_text = " cache hint and" if cache_hit else ""
        blue_print(f"{get_project_name()} Triton kernel{always_compile_text} saved at {os.path.dirname(metadata_path)}")

def check_dump_tensor_dtype(dump_tensor):
    import torch
    if dump_tensor.dtype != torch.float32:
        yellow_print(f"Warning: tensor dtype is {dump_tensor.dtype}, not torch.float32!")
