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
    result = subprocess.run([sys.executable, "examples/test.py"])
    if result.returncode == 0:
        passed.append(ver)
    else:
        failed.append(ver)

print(f"\n==========================================")
print(f"REGRESSION SUMMARY")
print(f"==========================================")
if passed:
    print(f"✅ PASS: {' '.join(passed)}")
if failed:
    print(f"❌ FAIL: {' '.join(failed)}")
    sys.exit(1)
