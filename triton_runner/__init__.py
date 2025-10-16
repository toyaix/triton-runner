__version__ = '0.2.7'

from .jit import jit
from .color_print import yellow_print
import os


def get_file_dir(file):
    return os.path.dirname(os.path.abspath(file))

try:
    import triton.language.extra.tlx as tlx
    is_tlx = True
except ImportError as e:
    is_tlx = False
