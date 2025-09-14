from .jit import jit

import os

os.environ["TRITON_ALWAYS_COMPILE"] = "1"

def get_file_dir(file):
    return os.path.dirname(os.path.abspath(file))
