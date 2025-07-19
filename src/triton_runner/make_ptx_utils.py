import re
import os
from triton._C.libtriton import llvm
from triton.backends.nvidia.compiler import get_ptx_version_from_options, sm_arch_from_capability, get_features


def runner_make_ptx(src, opt, capability):
    llvm.init_targets()
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
