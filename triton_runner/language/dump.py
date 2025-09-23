from ..jit import jit
import triton.language as tl
from triton._C.libtriton import ir
from typing import List

def dump(x):
    x.handle.set_attr("tt.dump", ir.make_attr([1], x.handle.get_context()))
