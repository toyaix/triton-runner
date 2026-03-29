import subprocess
import sys

# All supported versions. 3.3.x uses 3.3.1, 3.5.x uses 3.5.1.
ALL_VERSIONS = ["3.6.0", "3.5.1", "3.4.0", "3.3.1", "3.2.0", "3.1.0", "3.0.0"]

versions = sys.argv[1:] if len(sys.argv) > 1 else ALL_VERSIONS

passed = []
failed = []

for ver in versions:
    print(f"\n==========================================")
    print(f"Installing triton=={ver}")
    print(f"==========================================")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", f"triton=={ver}"], check=True)

    print(f"Running test on triton=={ver}...")
    proc = subprocess.Popen(
        [sys.executable, "examples/test.py"],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
    )
    fail_cmds = []
    collecting = False
    for line in proc.stdout:
        print(line, end="", flush=True)
        if "❌ SOME TEST FAIL" in line:
            collecting = True
            continue
        if collecting and line.strip():
            fail_cmds.append(line.strip())
    proc.wait()
    if proc.returncode == 0:
        passed.append(ver)
    else:
        failed.append((ver, fail_cmds))

summary_lines = []
summary_lines.append("==========================================")
summary_lines.append("REGRESSION SUMMARY")
summary_lines.append("==========================================")
if passed:
    summary_lines.append(f"✅ PASS: {' '.join(passed)}")
if failed:
    for ver, fail_cmds in failed:
        summary_lines.append(f"❌ FAIL: triton=={ver}")
        for cmd in fail_cmds:
            summary_lines.append(f"  - {cmd}")

summary = "\n".join(summary_lines)
print(f"\n{summary}")

with open("regression_result.txt", "w") as f:
    f.write(summary + "\n")

if failed:
    sys.exit(1)
