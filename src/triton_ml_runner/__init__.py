from .jit import jit

import os
def get_file_dir(file):
    return os.path.dirname(os.path.abspath(file))
