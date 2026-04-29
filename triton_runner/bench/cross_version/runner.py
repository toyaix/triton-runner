"""Cross-version benchmark runner — coordinates benchmarks across conda Triton environments.

Core workflow:
  1. (optional) Capture inputs once via :meth:`capture_inputs`
  2. Discover conda environments with different Triton versions
  3. In each env, compile & benchmark the user's kernel via ``conda run``
  4. Collect all result JSONs into a shared output directory
  5. Merge and generate comparison reports (terminal / CSV / Markdown)
"""

import json
import subprocess
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Sequence

from .problem_space import ProblemSpace


# ---- conda environment discovery ----

def discover_triton_envs(prefix: str = "triton-") -> dict[str, str]:
    """Find conda environments whose names start with *prefix* and have triton installed.

    Returns ``{env_name: triton_version}`` sorted by version.
    """
    try:
        result = subprocess.run(
            ["conda", "env", "list", "--json"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError("conda not available")
        env_data = json.loads(result.stdout)
    except (FileNotFoundError, json.JSONDecodeError, RuntimeError):
        return {}

    envs = {}
    for entry in env_data.get("envs", []):
        env_path = Path(entry)
        env_name = env_path.name
        if not env_name.startswith(prefix):
            continue

        ver = _get_env_triton_version(env_name)
        if ver:
            envs[env_name] = ver

    def _sort_key(item):
        ver = item[1].split(".")
        return tuple(int(v) for v in ver[:3])

    return dict(sorted(envs.items(), key=_sort_key))


def _get_env_triton_version(env_name: str) -> str | None:
    """Return the Triton version string for a conda env, or None."""
    try:
        result = subprocess.run(
            ["conda", "run", "-n", env_name, "python", "-c",
             "import triton; print(triton.__version__)"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return result.stdout.strip().split("\n")[-1]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return None


def _check_runner_in_env(env_name: str) -> bool:
    """Check if triton-runner is installed in the given conda env."""
    try:
        result = subprocess.run(
            ["conda", "run", "-n", env_name, "python", "-c",
             "import triton_runner"],
            capture_output=True, text=True, timeout=15,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


# ---- benchmark script template (runs inside each conda env) ----

_BENCH_TEMPLATE = r'''
import json, sys, importlib.util
from pathlib import Path
import torch, triton

kernel_path = {kernel_path!r}
output_path = {output_path!r}
sizes = {sizes!r}
warmup = {warmup}
rep = {rep}
lock_clocks = {lock_clocks!r}
inputs_dir = {inputs_dir!r}

spec = importlib.util.spec_from_file_location("_cross_version_kernel", kernel_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

# validate contract
_missing = []
for _attr in ("kernel", "prepare_args", "get_grid"):
    if not hasattr(mod, _attr) or not callable(getattr(mod, _attr)):
        _missing.append(_attr)
if _missing:
    raise AttributeError(
        f"Kernel module {{kernel_path!r}} missing: {{_missing}}. "
        f"See triton_runner.bench.cross_version.BenchmarkKernel for the contract."
    )

# ---- benchmark with raw CUDA events (works on all Triton versions) ----
def _bench_one(fn, warmup_ms, rep_ms):
    """Measure median kernel latency using CUDA events. No L2 cache trickery.

    This avoids ``triton.testing.do_bench`` / ``triton_runner.testing.do_bench``
    because those APIs differ across Triton versions (e.g. ``get_empty_cache_for_benchmark``
    doesn't exist before v3.2). Raw CUDA events work everywhere and are fair for A/B.
    """
    fn()
    torch.cuda.synchronize()

    # estimate single-run time
    _start = torch.cuda.Event(enable_timing=True)
    _end = torch.cuda.Event(enable_timing=True)
    _start.record()
    for _ in range(3):
        fn()
    _end.record()
    torch.cuda.synchronize()
    _est = _start.elapsed_time(_end) / 3

    _n_warmup = max(1, int(warmup_ms / _est)) if _est > 0 else 5
    _n_repeat = max(1, int(rep_ms / _est)) if _est > 0 else 100

    for _ in range(_n_warmup):
        fn()

    times = []
    for _ in range(10):
        _start = torch.cuda.Event(enable_timing=True)
        _end = torch.cuda.Event(enable_timing=True)
        _start.record()
        for _ in range(_n_repeat):
            fn()
        _end.record()
        torch.cuda.synchronize()
        times.append(_start.elapsed_time(_end) / _n_repeat)

    times.sort()
    return times[len(times) // 2]  # median


results = []
_cubin_output_dir = {cubin_output_dir!r}
if _cubin_output_dir:
    import os as _os, shutil as _shutil, tempfile as _tempfile
    _cubin_tmpdir = _tempfile.mkdtemp(prefix="triton_cubin_")
    _os.environ["TRITON_CACHE_DIR"] = _cubin_tmpdir
for i, size in enumerate(sizes):
    if inputs_dir:
        inputs_file = Path(inputs_dir) / f"{{i:04d}}.pt"
        args = torch.load(str(inputs_file), weights_only=False)
    else:
        args = mod.prepare_args(**size)

    grid = mod.get_grid(**size)
    kernel_kwargs = mod.get_kernel_kwargs(**size) if hasattr(mod, "get_kernel_kwargs") else {{}}

    mod.kernel[grid](*args, **kernel_kwargs)
    torch.cuda.synchronize()

    # Save all compiled artifacts from the temp cache after first compilation.
    # This includes .cubin, .json (metadata), .ptx, .ttir, .ttgir, .llir —
    # everything needed for cubin_dir / ptx_src / ttgir_dir etc. reproduction.
    if _cubin_output_dir and i == 0:
        _artifacts = []
        if _os.path.isdir(_cubin_tmpdir):
            for _r, _ds, _fs in _os.walk(_cubin_tmpdir):
                for _f in _fs:
                    _artifacts.append(_os.path.join(_r, _f))
        if _artifacts:
            _os.makedirs(_cubin_output_dir, exist_ok=True)
            for _ap in _artifacts:
                _shutil.copy2(_ap, _os.path.join(_cubin_output_dir, _os.path.basename(_ap)))
            _n_cubin = sum(1 for a in _artifacts if a.endswith(".cubin"))
            print(f"  [cubin] saved {{_n_cubin}} cubin(s) + {{len(_artifacts) - _n_cubin}} metadata/IR file(s) to {{_cubin_output_dir}}", file=sys.stderr)
        else:
            print(f"  [cubin] WARNING: no artifacts produced in {{_cubin_tmpdir}}", file=sys.stderr)

    fn = lambda: mod.kernel[grid](*args, **kernel_kwargs)
    latency = _bench_one(fn, warmup, rep)

    results.append({{
        "index": i,
        "problem_size": size,
        "latency_ms": round(latency, 6),
        "grid": list(grid),
        "triton_version": triton.__version__,
        "pytorch_version": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
        "timestamp": __import__("datetime").datetime.now().isoformat(),
        "replayed_inputs": bool(inputs_dir),
    }})
    print(f"  [{{i}}] {{size}}: {{latency:.4f}} ms", file=sys.stderr)

output = {{
    "kernel": kernel_path,
    "results": results,
}}
Path(output_path).write_text(json.dumps(output, indent=2, ensure_ascii=False))
print(f"Saved to {{output_path}}", file=sys.stderr)

# Clean up temp cache directory
if _cubin_output_dir:
    _shutil.rmtree(_cubin_tmpdir, ignore_errors=True)
'''


# ---- input capture script (runs in the *host* environment) ----

_CAPTURE_TEMPLATE = r'''
import importlib.util, json, sys
from pathlib import Path
import torch

kernel_path = {kernel_path!r}
sizes = {sizes!r}
inputs_dir = {inputs_dir!r}

spec = importlib.util.spec_from_file_location("_cap_kernel", kernel_path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

out = Path(inputs_dir)
out.mkdir(parents=True, exist_ok=True)

for i, size in enumerate(sizes):
    args = mod.prepare_args(**size)
    # args is a tuple of tensors / scalars — store only positional args,
    # get_kernel_kwargs values are constant and don't need capturing.
    filepath = out / f"{{i:04d}}.pt"
    torch.save(args, str(filepath))
    print(f"  captured [{{i}}] {{size}} -> {{filepath}}", file=sys.stderr)

# write metadata for reproducibility
meta = {{
    "kernel_path": kernel_path,
    "sizes": sizes,
    "num_inputs": len(sizes),
}}
(out / "metadata.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
print(f"Captured {{len(sizes)}} input(s) to {{inputs_dir}}", file=sys.stderr)
'''


# ---- main runner class ----

class CrossVersionRunner:
    """Coordinate benchmarks across multiple conda Triton environments.

    Usage::

        runner = CrossVersionRunner(
            kernel="kernels/matmul.py",
            output_dir="cross_version_results/",
        )
        runner.capture_inputs()              # generate + save inputs once
        runner.setup_envs()
        runner.run()                         # all envs use same inputs
        runner.compare(output_md="report.md", output_csv="report.csv")
    """

    def __init__(
        self,
        kernel: str,
        output_dir: str = "cross_version_results",
        envs: Sequence[str] | None = None,
        env_prefix: str = "triton-",
        sizes: list[dict] | ProblemSpace | str | None = None,
        warmup: int = 25,
        rep: int = 100,
        lock_clocks: tuple[int, int] | None = None,
        save_cubin: bool = False,
    ):
        self.kernel = str(Path(kernel).resolve())
        self.output_dir = Path(output_dir).resolve()
        self.envs = list(envs) if envs else []
        self.env_prefix = env_prefix
        self.warmup = warmup
        self.rep = rep
        self.lock_clocks = lock_clocks
        self.save_cubin = save_cubin

        self._sizes_raw = sizes
        self.sizes: list[dict] = []
        self._problem_space: ProblemSpace | None = None

        self._env_versions: dict[str, str] = {}
        self._result_paths: dict[str, Path] = {}
        self._all_results: list[dict] = []

    # ---- problem space resolution ----

    def _resolve_sizes(self) -> tuple[list[dict], ProblemSpace | None]:
        raw = self._sizes_raw

        if isinstance(raw, list):
            return raw, None

        if isinstance(raw, ProblemSpace):
            self._problem_space = raw
            return raw.generate(), raw

        if isinstance(raw, str):
            path = Path(raw)
            if path.suffix == ".json":
                ps = ProblemSpace.from_json(str(path))
                if ps is not None:
                    self._problem_space = ps
                    return ps.generate(), ps
            ps = ProblemSpace.from_kernel(str(path))
            if ps is not None:
                self._problem_space = ps
                return ps.generate(), ps
            return self._default_sizes(), None

        ps = ProblemSpace.from_kernel(self.kernel)
        if ps is not None:
            self._problem_space = ps
            return ps.generate(), ps

        return self._default_sizes(), None

    def _default_sizes(self) -> list[dict]:
        sizes = [256, 512, 1024, 2048, 4096]
        return [{"M": s, "N": s, "K": s} for s in sizes]

    @property
    def problem_space(self) -> ProblemSpace | None:
        return self._problem_space

    @property
    def inputs_dir(self) -> Path:
        return self.output_dir / "inputs"

    @property
    def has_captured_inputs(self) -> bool:
        """True if ``output_dir/inputs/`` already contains captured data."""
        meta = self.inputs_dir / "metadata.json"
        return meta.exists()

    # ---- input capture ----

    def capture_inputs(self, sizes: list[dict] | None = None) -> Path:
        """Generate inputs once in the **current** environment and save to disk.

        Calls ``prepare_args(**size)`` for each problem size, serializes the
        tensor tuple via ``torch.save`` into ``output_dir/inputs/``.

        Subsequent :meth:`run` calls automatically detect and replay these
        saved inputs, guaranteeing that every Triton version benchmarks
        against the **exact same** input data.
        """
        sizes = sizes or self.sizes
        if not sizes:
            self.sizes, _ = self._resolve_sizes()
            sizes = self.sizes

        indir = self.inputs_dir
        indir.mkdir(parents=True, exist_ok=True)

        script = _CAPTURE_TEMPLATE.format(
            kernel_path=self.kernel,
            sizes=sizes,
            inputs_dir=str(indir),
        )

        print(f"[capture] Generating {len(sizes)} input(s) ...")
        result = subprocess.run(
            ["python", "-c", script],
            capture_output=True, text=True, timeout=120,
        )
        if result.returncode != 0:
            print(f"  FAILED: {result.stderr}", file=sys.stderr)
            raise RuntimeError(f"Input capture failed: {result.stderr}")
        if result.stderr:
            print(result.stderr.strip())

        print(f"  -> {indir}/ ({len(sizes)} .pt files + metadata.json)")
        return indir

    # ---- discovery ----

    def discover_envs(self) -> dict[str, str]:
        if self.envs:
            result = {}
            for name in self.envs:
                ver = _get_env_triton_version(name)
                if ver:
                    result[name] = ver
            return result
        return discover_triton_envs(prefix=self.env_prefix)

    def setup_envs(self):
        env_versions = self.discover_envs()
        if not env_versions:
            raise RuntimeError(
                f"No conda Triton environments found (prefix='{self.env_prefix}'). "
                f"Create them first, e.g.: conda create -n triton-v3-7-0 python=3.10 && "
                f"conda run -n triton-v3-7-0 pip install triton triton-runner"
            )

        for env_name, ver in env_versions.items():
            if not _check_runner_in_env(env_name):
                print(f"[setup] Installing triton-runner in {env_name} ...")
                subprocess.run(
                    ["conda", "run", "-n", env_name, "pip", "install", "triton-runner"],
                    check=True,
                )
        self._env_versions = env_versions
        print(f"[setup] Ready: {len(env_versions)} environment(s)")
        for name, ver in env_versions.items():
            print(f"  {name}: Triton {ver}")

        # resolve problem space
        self.sizes, ps = self._resolve_sizes()
        if ps is not None:
            print(f"[sizes] {ps.size} config(s) via {ps.mode!r} mode: "
                  f"{list(ps.dimensions.keys())}")
        else:
            print(f"[sizes] {len(self.sizes)} config(s) (inline list)")

        # detect captured inputs
        if self.has_captured_inputs:
            print(f"[inputs] Replay mode — using captured inputs from {self.inputs_dir}/")

    # ---- run ----

    def run(self, sizes: list[dict] | None = None) -> dict[str, Path]:
        """Run the kernel benchmark in each Triton environment.

        If ``output_dir/inputs/`` exists (from a prior :meth:`capture_inputs`),
        each environment replays those pre-serialized tensors instead of
        calling ``prepare_args()``.  This guarantees identical input data
        across all Triton versions.

        Returns ``{env_name: Path_to_json}``.
        """
        if not self._env_versions:
            self.setup_envs()

        sizes = sizes or self.sizes
        if not sizes:
            self.sizes, _ = self._resolve_sizes()
            sizes = self.sizes
        self.output_dir.mkdir(parents=True, exist_ok=True)

        for env_name, ver in self._env_versions.items():
            output_path = self.output_dir / f"{env_name}.json"
            lc = list(self.lock_clocks) if self.lock_clocks else None

            cubin_output_dir = ""
            if self.save_cubin:
                cubin_output_dir = str(self.output_dir / "cubins" / env_name)

            script = _BENCH_TEMPLATE.format(
                kernel_path=self.kernel,
                output_path=str(output_path),
                sizes=sizes,
                warmup=self.warmup,
                rep=self.rep,
                lock_clocks=lc,
                inputs_dir=str(self.inputs_dir) if self.has_captured_inputs else "",
                cubin_output_dir=cubin_output_dir,
            )

            tag = "(replay)" if self.has_captured_inputs else "(live)"
            print(f"[bench] {env_name} (Triton {ver}) {tag} ...")
            result = subprocess.run(
                ["conda", "run", "-n", env_name, "python", "-c", script],
                capture_output=True, text=True, timeout=600,
            )
            if result.returncode != 0:
                print(f"  FAILED: {result.stderr}", file=sys.stderr)
                continue
            if result.stderr:
                print(result.stderr.strip())

            self._result_paths[env_name] = output_path
            print(f"  -> {output_path}")

        return self._result_paths

    # ---- compare ----

    def compare(
        self,
        output_csv: str = "",
        output_md: str = "",
        print_table: bool = True,
    ) -> str:
        """Merge collected results and generate a comparison report."""
        if not self._result_paths:
            self._load_existing_results()
        else:
            self._load_existing_results()

        if len(self._all_results) < 2:
            return "[compare] Need at least 2 result sets to compare."

        return self._generate_report(
            output_csv=str(self.output_dir / output_csv) if output_csv else "",
            output_md=str(self.output_dir / output_md) if output_md else "",
            print_table=print_table,
        )

    # ---- internals ----

    def _load_existing_results(self):
        self._all_results = []
        self._result_paths = {}
        for f in sorted(self.output_dir.glob("*.json")):
            data = json.loads(f.read_text())
            self._all_results.append(data)
            self._result_paths[f.stem] = f

    def _generate_report(
        self, output_csv: str, output_md: str, print_table: bool
    ) -> str:
        by_size: dict[tuple, dict] = defaultdict(dict)
        all_versions: set[str] = set()
        replayed = False

        for data in self._all_results:
            for entry in data["results"]:
                key = tuple(sorted(entry["problem_size"].items()))
                by_size[key][entry["triton_version"]] = entry
                all_versions.add(entry["triton_version"])
                if entry.get("replayed_inputs"):
                    replayed = True

        sorted_versions = sorted(all_versions)
        baseline_ver = sorted_versions[0]

        table_rows: list[dict] = []
        lines: list[str] = []

        header_cols = ["Problem"] + [f"{v} (ms)" for v in sorted_versions] + ["Speedup"]
        header = " | ".join(f"{h:>14}" for h in header_cols)
        sep = "-" * len(header)
        lines.append(header)
        lines.append(sep)

        for size_key in sorted(by_size):
            vr = by_size[size_key]
            if baseline_ver not in vr:
                continue
            baseline = vr[baseline_ver]["latency_ms"]
            latencies = []
            for v in sorted_versions:
                latencies.append(vr[v]["latency_ms"] if v in vr else float("nan"))

            speedup = baseline / latencies[-1] if latencies[-1] > 0 else float("nan")
            size_str = str(dict(size_key))
            cols = [size_str] + [f"{l:.4f}" for l in latencies] + [f"{speedup:.2f}x"]
            lines.append(" | ".join(f"{c:>14}" for c in cols))

            table_rows.append({
                "problem": size_str,
                **{v: l for v, l in zip(sorted_versions, latencies)},
                "speedup": round(speedup, 3),
            })

        table = "\n".join(lines)

        if print_table:
            if replayed:
                print("[inputs were replayed from capture — same data across all versions]")
            print(table)

        if output_csv:
            import csv
            with open(output_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=table_rows[0].keys())
                writer.writeheader()
                writer.writerows(table_rows)
            print(f"\nCSV -> {output_csv}")

        if output_md:
            replay_note = (" (identical inputs replayed across all versions)"
                           if replayed else "")
            md_lines = [
                "# Cross-Version Performance Report",
                "",
                f"**Kernel:** `{self._all_results[0].get('kernel', 'N/A')}`",
                f"**GPU:** {self._all_results[0]['results'][0].get('gpu', 'N/A')}",
                f"**Versions:** {', '.join(sorted_versions)}",
                f"**Date:** {datetime.now().isoformat()}",
                f"**Inputs:** {'replayed' if replayed else 'live-generated'}{replay_note}",
                "",
                "## Results",
                "",
                "| Problem | " + " | ".join(f"**{v}**" for v in sorted_versions) + " | Speedup |",
                "|---------|" + "|".join(["-" * 10] * (len(sorted_versions) + 1)) + "|",
            ]
            for row in table_rows:
                vs = " | ".join(f"{row[v]:.4f}" for v in sorted_versions)
                md_lines.append(
                    f"| {row['problem']} | {vs} | {row['speedup']:.2f}x |"
                )
            md_lines.append("")
            md_lines.append(f"*Baseline: {baseline_ver}. Timing: raw CUDA events, median-of-10, warmup={self.warmup}ms rep={self.rep}ms.*")
            md_lines.append("")
            Path(output_md).write_text("\n".join(md_lines) + "\n")
            print(f"MD  -> {output_md}")

        return table
