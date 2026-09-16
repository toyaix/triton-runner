"""CPU-only checks for the MLIR dump ownership state machine.

Covers parse_mlir_to_folder's manifest bookkeeping: stale-file cleanup, empty
parses, and pre-existing files/symlinks at the bookkeeping paths. No GPU or
Triton compilation is required, so this runs in plain CI.
"""
import json
import os
import sys
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from triton_runner.compiler.compile import (  # noqa: E402
    _MLIR_MANIFEST_NAME,
    _manifest_names,
    _pick_output_folder,
    _write_manifest,
    parse_mlir_to_folder,
)

DUMP = "// -----// IR Dump Before {pass_name} (ttir) (op) //----- //\nmodule {{}}\n"


def _mlir(markers):
    return "".join(DUMP.format(pass_name=name) for name in markers)


def _run_parse(cache_dir, markers, file_name="all.mlir"):
    os.environ["MLIR_ENABLE_DUMP"] = "1"
    path = Path(cache_dir) / file_name
    path.write_text(_mlir(markers))
    parse_mlir_to_folder(str(path))


def _state(cache_dir):
    folder = Path(cache_dir) / "mlir"
    files = sorted(p.name for p in folder.iterdir() if p.name != _MLIR_MANIFEST_NAME)
    return files, sorted(_manifest_names(str(folder)))


def test_stale_files_removed_and_folder_reused():
    with tempfile.TemporaryDirectory() as cache:
        _run_parse(cache, ["passa", "passb"])  # 01-source, 02-passa, 03-passb
        before, _ = _state(cache)
        assert before == ["01-source.mlir", "02-passa.mlir", "03-passb.mlir"], before

        _run_parse(cache, ["passa"])  # 03-passb becomes stale
        files, manifest = _state(cache)
        assert files == ["01-source.mlir", "02-passa.mlir"], files
        assert manifest == files, manifest
        assert not os.path.exists(os.path.join(cache, "mlir-1")), "folder must stay reusable"

        _run_parse(cache, ["passa"])  # rerun with the same output set
        assert not os.path.exists(os.path.join(cache, "mlir-1")), "folder must stay reusable"


def test_empty_parse_preserves_previous_dump():
    with tempfile.TemporaryDirectory() as cache:
        _run_parse(cache, ["passa"])
        files, manifest = _state(cache)
        os.unlink(os.path.join(cache, "all.mlir"))
        _run_parse(cache, [])  # torn/crash-only dump: no markers
        assert _state(cache) == (files, manifest), "empty parse must touch nothing"
        assert not os.path.exists(os.path.join(cache, "mlir-1"))


def test_manifest_tmp_path_symlink_is_not_followed():
    with tempfile.TemporaryDirectory() as cache:
        folder = Path(cache) / "mlir"
        folder.mkdir()
        victim = Path(cache) / "victim.txt"
        victim.write_text("do not touch")
        os.symlink(victim, folder / f"{_MLIR_MANIFEST_NAME}.tmp")
        _run_parse(cache, ["passa"])
        assert victim.read_text() == "do not touch"
        files, manifest = _state(cache)
        planted = f"{_MLIR_MANIFEST_NAME}.tmp"  # foreign leftovers stay, are never read
        assert sorted(n for n in files if n != planted) == manifest == ["01-source.mlir", "02-passa.mlir"]


def test_dangling_manifest_symlink_is_safe():
    with tempfile.TemporaryDirectory() as cache:
        folder = Path(cache) / "mlir"
        folder.mkdir()
        os.symlink(os.path.join(cache, "missing-target"), folder / _MLIR_MANIFEST_NAME)
        _run_parse(cache, ["passa"])
        files, manifest = _state(cache)
        assert manifest == files, "dangling manifest link must heal, not crash"
        assert not os.path.islink(folder / _MLIR_MANIFEST_NAME)


def test_crash_after_claim_still_reuses_folder():
    # simulate a crash after the union manifest is written but before the
    # dump files land: the folder must stay owned and self-heal on rerun
    with tempfile.TemporaryDirectory() as cache:
        _run_parse(cache, ["passa"])
        folder = Path(cache) / "mlir"
        _write_manifest(str(folder), ["01-source.mlir", "02-passa.mlir", "03-other.mlir"])
        assert _pick_output_folder(cache, ["01-source.mlir", "02-passa.mlir"]) == str(folder)
        _run_parse(cache, ["passa"])
        files, manifest = _state(cache)
        assert files == manifest == ["01-source.mlir", "02-passa.mlir"]


def main():
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
        print(f"✅ {test.__name__}")
    print(f"ALL {len(tests)} MLIR MANIFEST TESTS PASS")


if __name__ == "__main__":
    main()
