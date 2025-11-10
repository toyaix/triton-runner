import triton
import torch
import os
import re
import subprocess
import triton_runner

triton_version = triton.__version__
if triton_version in ["3.3.0", "3.3.1"]:
    triton_version = "3.3.x"

def get_content(file_path):
    return open(file_path, "r").read()

def get_lines(match):
    return [line.strip() for line in match.group(1).strip().splitlines() if line.strip()]

device = torch.cuda.current_device()
capability = torch.cuda.get_device_capability(device)
capability = capability[0] * 10 + capability[1]

pattern = re.compile(rf"### sm{capability}.*?shell(.*?)```", re.DOTALL)
runner_file_path = os.path.join("examples", "runner", f"v{triton_version}", "README.md")
match = pattern.search(get_content(runner_file_path))
if match:
    lines = get_lines(match)
    pattern = re.compile(rf"shell(.*?)```", re.DOTALL)
    bench_file_path = os.path.join("doc", "benchmark.md")
    lines.extend(get_lines(pattern.search(get_content(bench_file_path))))
    if triton_version in ["3.3.x", "3.4.0", "3.5.0"]:
        debug_file_path = os.path.join("doc", "dump.md")
        for i, m in enumerate(pattern.finditer((get_content(debug_file_path)), 1)):
            lines.extend(get_lines(m))
    triton_runner.color_print.yellow_print(f"TEST on triton v{triton.__version__}")
    fail_cmd = []
    for cmd in lines:
        triton_runner.color_print.blue_print(cmd)
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        print("stdout:", result.stdout)
        print("return code:", result.returncode)
        if result.returncode:
            fail_cmd.append(cmd)
    if len(fail_cmd) == 0:
        print(f"✅ ALL TEST PASS on triton v{triton.__version__}")
    else:
        triton_runner.color_print.yellow_print(f"❌ SOME TEST FAIL on triton v{triton.__version__}")
        print("\n".join(fail_cmd))
else:
    print(f"sm{capability} on triton v{triton.__version__} not found")
