import re
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
SELF_PATH = Path(__file__).resolve()
SCAN_PATHS = (
    "README.md",
    "doc",
    "examples",
    "skills",
    "test",
)
TEXT_SUFFIXES = {".md", ".py"}
EXAMPLE_CMD_RE = re.compile(r"python\s+(examples/[^\s\"'`]+?\.py)")


def iter_text_files():
    for relative_path in SCAN_PATHS:
        path = REPO_ROOT / relative_path
        if path.is_file():
            if path.suffix in TEXT_SUFFIXES and path.resolve() != SELF_PATH:
                yield path
            continue

        for candidate in path.rglob("*"):
            if not candidate.is_file():
                continue
            if "__pycache__" in candidate.parts:
                continue
            if candidate.suffix not in TEXT_SUFFIXES:
                continue
            if candidate.resolve() == SELF_PATH:
                continue
            yield candidate


def collect_example_command_refs():
    refs = {}
    for path in iter_text_files():
        for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1):
            for match in EXAMPLE_CMD_RE.finditer(line):
                cmd_path = match.group(1)
                refs.setdefault(cmd_path, []).append((path.relative_to(REPO_ROOT), lineno))
    return refs


def main():
    refs = collect_example_command_refs()
    missing = {
        cmd_path: locations
        for cmd_path, locations in refs.items()
        if not (REPO_ROOT / cmd_path).exists()
    }

    if not missing:
        print(f"checked {len(refs)} unique python examples commands: all referenced files exist")
        return

    print("missing files referenced by `python examples/...py` commands:")
    for cmd_path in sorted(missing):
        print(cmd_path)
        for source_path, lineno in missing[cmd_path]:
            print(f"  {source_path}:{lineno}")
    sys.exit(1)


if __name__ == "__main__":
    main()
