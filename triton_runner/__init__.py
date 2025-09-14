from .jit import jit
from .color_print import yellow_print
import os


if os.environ.get("TRITON_ALWAYS_COMPILE", "1") == "1":
    yellow_print("[triton-runner] TRITON_ALWAYS_COMPILE defaults to ON.\nFor production, It is recommended to disable it by running: export TRITON_ALWAYS_COMPILE=0")


def get_file_dir(file):
    return os.path.dirname(os.path.abspath(file))
