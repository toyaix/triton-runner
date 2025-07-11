import tempfile
import subprocess
import os
import triton
import signal
import re
from .check_utils import check_cuda_arch_with_capability
from triton.backends.nvidia.compiler import get_ptxas, sm_arch_from_capability


def runner_make_cubin(src, opt, capability):
    ptxas, _ = get_ptxas(capability)
    with tempfile.NamedTemporaryFile(delete=False, mode='w', suffix='.ptx') as fsrc, \
        tempfile.NamedTemporaryFile(delete=False, mode='r', suffix='.log') as flog:
        fsrc.write(src)
        fsrc.flush()
        fbin = fsrc.name + '.o'

        line_info = ["-lineinfo", "-suppress-debug-info"] if os.environ.get("TRITON_DISABLE_LINE_INFO",
                                                                            "0") == "1" else ["-lineinfo"]
        fmad = [] if opt.enable_fp_fusion else ['--fmad=false']
        arch = sm_arch_from_capability(capability)
        opt_level = ['--opt-level', '0'] if os.environ.get("DISABLE_PTXAS_OPT", "0") == "1" else []
        ptxas_cmd = [ptxas, *line_info, *fmad, '-v', *opt_level, f'--gpu-name={arch}', fsrc.name, '-o', fbin]
        try:
            subprocess.run(ptxas_cmd, check=True, close_fds=False, stderr=flog)
            if os.path.exists(fsrc.name):
                os.remove(fsrc.name)
            if os.path.exists(flog.name):
                os.remove(flog.name)
        except subprocess.CalledProcessError as e:
            with open(flog.name) as log_file:
                log = log_file.read()
            if os.path.exists(flog.name):
                os.remove(flog.name)

            if e.returncode == 255:
                error = 'Internal Triton PTX codegen error'
            elif e.returncode == 128 + signal.SIGSEGV:
                error = '`ptxas` raised SIGSEGV'
            else:
                error = f'`ptxas` failed with error code {e.returncode}'

            raise triton.runtime.errors.PTXASError(f"{error}\n"
                                                   f"`ptxas` stderr:\n{log}\n"
                                                   f'Repro command: {" ".join(ptxas_cmd)}\n')

        with open(fbin, 'rb') as f:
            cubin = f.read()
        if os.path.exists(fbin):
            os.remove(fbin)
    return cubin


def get_cubin(full_path, kernel_name, save_path):
    target = triton.runtime.driver.active.get_current_target()
    backend = triton.compiler.make_backend(target)
    options = backend.parse_options(dict())
    src = open(full_path, 'r').read()
    target_match = re.search(r"\.target\s+sm_(\d+)", src)
    ptx_capability = target_match.group(1)
    check_cuda_arch_with_capability(ptx_capability, target.arch)
    cubin = runner_make_cubin(src, options, target.arch)
    cubin_path = os.path.join(save_path, f"{kernel_name}.cubin")
    print(cubin_path)
    with open(cubin_path, "wb") as cubin_file:
        cubin_file.write(cubin)
