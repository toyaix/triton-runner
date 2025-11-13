__version__ = '0.3.1'

from .jit import jit
from . import color_print
from . import torch_utils
import os


def get_file_dir(file):
    return os.path.dirname(os.path.abspath(file))
