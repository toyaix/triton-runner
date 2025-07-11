import re
import os
import triton
from triton._C.libtriton import llvm
from triton.backends.nvidia.compiler import get_ptx_version_from_options, sm_arch_from_capability, get_features
from .make_cubin_utils import save_cubin_from_ptx

def runner_make_ptx(src, opt, capability):
    ptx_version = get_ptx_version_from_options(opt, capability)

    triple = 'nvptx64-nvidia-cuda'
    proc = sm_arch_from_capability(capability)
    features = get_features(opt, capability)
    ret = llvm.translate_to_asm(src, triple, proc, features, ['nvptx-short-ptr'], opt.enable_fp_fusion, False)
    # Find kernel names (there should only be one)
    names = re.findall(r".visible .entry ([a-zA-Z_][a-zA-Z0-9_]*)", ret)
    assert len(names) == 1
    # post-process
    ptx_version = f'{ptx_version//10}.{ptx_version%10}'
    ret = re.sub(r'\.version \d+\.\d+', f'.version {ptx_version}', ret, flags=re.MULTILINE)
    ret = re.sub(r'\.target sm_\d+', f'.target sm_{capability}', ret, flags=re.MULTILINE)
    # Remove the debug flag that prevents ptxas from optimizing the code
    ret = re.sub(r",\s*debug|debug,\s*", "", ret)
    if os.environ.get("NVPTX_ENABLE_DUMP", "0") == "1":
        print("// -----// NVPTX Dump //----- //")
        print(ret)
    return ret

def save_cubin_from_llir(full_path, kernel_name, save_path):
    target = triton.runtime.driver.active.get_current_target()
    backend = triton.compiler.make_backend(target)
    options = backend.parse_options(dict())
    src = open(full_path, 'r').read()
    llvm.init_targets()
    ptx = runner_make_ptx(src, options, target.arch)
    ptx_path = os.path.join(save_path, f"{kernel_name}.ptx")
    with open(ptx_path, "w") as ptx_file:
        ptx_file.write(ptx)
    save_cubin_from_ptx(ptx_path, kernel_name, save_path)
