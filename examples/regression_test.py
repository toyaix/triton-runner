import subprocess
import sys

# All supported versions. 3.3.x uses 3.3.1, 3.5.x uses 3.5.1.
ALL_VERSIONS = ["3.0.0", "3.1.0", "3.2.0", "3.3.1", "3.4.0", "3.5.1", "3.6.0"]

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

print(f"\n==========================================")
print(f"REGRESSION SUMMARY")
print(f"==========================================")
if passed:
    print(f"✅ PASS: {' '.join(passed)}")
if failed:
    for ver, fail_cmds in failed:
        print(f"❌ FAIL: triton=={ver}")
        for cmd in fail_cmds:
            print(f"  - {cmd}")
    sys.exit(1)
