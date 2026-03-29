from __future__ import annotations

import functools
import os
import subprocess
from pathlib import Path
from typing import Any, Callable

from triton import knobs
from triton.runtime.build import _build, _load_module_from_path


def _dedupe_paths(paths: list[str]) -> tuple[str, ...]:
    unique_paths: list[str] = []
    for path in paths:
        if path and path not in unique_paths:
            unique_paths.append(path)
    return tuple(unique_paths)


@functools.lru_cache()
def libcuda_dirs() -> tuple[str, ...]:
    if env_libcuda_path := knobs.nvidia.libcuda_path:
        return (env_libcuda_path,)

    locs: list[str] = []
    try:
        libs = subprocess.check_output(["/sbin/ldconfig", "-p"]).decode(errors="ignore")
        locs = [line.split()[-1] for line in libs.splitlines() if "libcuda.so.1" in line]
    except Exception:
        pass

    dirs = [os.path.dirname(loc) for loc in locs]
    env_ld_library_path = os.getenv("LD_LIBRARY_PATH")
    if env_ld_library_path and not dirs:
        dirs = [path for path in env_ld_library_path.split(":") if os.path.exists(os.path.join(path, "libcuda.so.1"))]

    msg = "libcuda.so cannot found!\n"
    if locs:
        msg += f"Possible files are located at {locs}."
        msg += "Please create a symlink of libcuda.so to any of the files."
    else:
        msg += 'Please make sure GPU is set up and then run "/sbin/ldconfig"'
        msg += " (requires sudo) to refresh the linker cache."
    assert any(os.path.exists(os.path.join(path, "libcuda.so.1")) for path in dirs), msg
    return _dedupe_paths(dirs)


@functools.lru_cache()
def cuda_home() -> Path:
    try:
        from tvm_ffi.cpp.extension import _find_cuda_home  # type: ignore[attr-defined]

        return Path(_find_cuda_home())
    except Exception:
        return Path("/usr/local/cuda")


@functools.lru_cache()
def cuda_include_dirs() -> tuple[str, ...]:
    include_dir = cuda_home() / "include"
    if not include_dir.is_dir():
        return ()
    return (str(include_dir),)


@functools.lru_cache()
def cuda_library_dirs(*, include_stubs: bool = False) -> tuple[str, ...]:
    candidates = [
        cuda_home() / "lib64",
        cuda_home() / "lib",
    ]
    if include_stubs:
        candidates.extend([
            cuda_home() / "lib64" / "stubs",
            cuda_home() / "lib" / "stubs",
        ])

    dirs = [str(path) for path in candidates if path.is_dir()]
    dirs.extend(libcuda_dirs())
    return _dedupe_paths(dirs)


def library_dirs(*extra_dirs: str | Path, include_stubs: bool = False) -> tuple[str, ...]:
    dirs = [str(path) for path in extra_dirs if path]
    dirs.extend(cuda_library_dirs(include_stubs=include_stubs))
    return _dedupe_paths(dirs)


def build_module_from_src(
        *,
        module_name: str,
        build_dir: str | Path,
        source: str,
        include_dirs: tuple[str, ...] | list[str] = (),
        library_dirs: tuple[str, ...] | list[str] = (),
        libraries: tuple[str, ...] | list[str] = (),
        ccflags: tuple[str, ...] | list[str] = (),
        source_ext: str = ".c",
        final_path: str | Path | None = None,
        load_module: Callable[[str], Any] | None = None,
) -> Any:
    build_dir = Path(build_dir)
    build_dir.mkdir(parents=True, exist_ok=True)
    source_path = build_dir / f"{module_name}{source_ext}"
    source_path.write_text(source)

    built_path = Path(
        _build(
            module_name,
            str(source_path),
            str(build_dir),
            list(library_dirs),
            list(include_dirs),
            list(libraries),
            list(ccflags),
        ))
    final_path = built_path if final_path is None else Path(final_path)
    if built_path != final_path:
        final_path.unlink(missing_ok=True)
        built_path.replace(final_path)
    if load_module is None:
        return _load_module_from_path(module_name, str(final_path))
    return load_module(str(final_path))
