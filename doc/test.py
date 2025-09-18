import triton
import torch
import os
import re
import subprocess
import triton_runner

triton_version = triton.__version__
if triton_version in ["3.3.0", "3.3.1"]:
    triton_version = "3.3.x"

def get_content(file_name):
    doc_file_path = os.path.join("doc", file_name)
    return open(doc_file_path, "r").read()

def get_lines(match):
    return [line.strip() for line in match.group(1).strip().splitlines() if line.strip()]

device = torch.cuda.current_device()
capability = torch.cuda.get_device_capability(device)
capability = capability[0] * 10 + capability[1]

pattern = re.compile(rf"### sm{capability}.*?shell(.*?)```", re.DOTALL)

match = pattern.search(get_content(f"examples_v{triton_version}.md"))
if match:
    lines = get_lines(match)
    pattern = re.compile(rf"shell(.*?)```", re.DOTALL)
    lines.extend(get_lines(pattern.search(get_content("benchmark.md"))))
    if triton_version in ["3.4.0"]:
        lines.extend(get_lines(pattern.search(get_content("debug_tool.md"))))
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
