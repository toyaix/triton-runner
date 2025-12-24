__version__ = '0.3.3'

from .version_utils import is_support_version, triton_version
if not is_support_version:
    raise RuntimeError(f"Triton Runner doesn't support Triton v{triton_version}")


from .jit import jit
from .version_utils import is_triton_geq_v3_4
if is_triton_geq_v3_4:
    from .autotune import autotune
from . import color_print
from . import torch_utils
import os


def get_file_dir(file):
    return os.path.dirname(os.path.abspath(file))
