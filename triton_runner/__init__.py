__version__ = '0.3.4'

from .version_utils import is_support_version, triton_version
if not is_support_version:
    raise RuntimeError(f"Triton Runner doesn't support Triton v{triton_version}")


def _env_flag(name, default=True):
    import os
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "off", "no", ""}

TRITON_TVM_FFI = _env_flag("TRITON_TVM_FFI", default=False)


from .jit import jit
from .version_utils import is_triton_geq_v3_4
if is_triton_geq_v3_4:
    from .autotune import autotune
from . import color_print
from . import torch_utils

def get_file_dir(file):
    import os
    return os.path.dirname(os.path.abspath(file))
