import os

def list_files_rel(path):
    files = []
    for root, _, filenames in os.walk(path):
        for f in filenames:
            if f.endswith('.py'):
                rel_path = os.path.relpath(os.path.join(root, f), path)
                files.append(rel_path)
    return sorted(files)

old_dirname = ""
for f in list_files_rel("examples/debugging/python"):
    dirname, filename = os.path.split(f)
    if dirname != old_dirname:
        print()
    old_dirname = dirname
    print("python", f"examples/debugging/python/{f}")
