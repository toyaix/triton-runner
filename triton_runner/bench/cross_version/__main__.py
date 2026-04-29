"""CLI entry point for cross-version benchmarking.

Usage::

    # Auto-discover conda envs, auto-detect kernel's get_problem_space()
    python -m triton_runner.bench.cross_version -k kernels/matmul.py

    # Specify envs explicitly
    python -m triton_runner.bench.cross_version -k kernels/matmul.py -e triton-v3-5-0 triton-v3-7-0

    # Load problem space from JSON
    python -m triton_runner.bench.cross_version -k kernels/matmul.py --sizes-json sizes.json

    # Quick sweep: vary M, fix N=K=1024
    python -m triton_runner.bench.cross_version -k kernels/matmul.py \\
        --sweep M -v 256 512 1024 2048 4096 --fixed N=1024 K=1024

    # Cartesian product mode
    python -m triton_runner.bench.cross_version -k kernels/matmul.py \\
        --product -d M 256 1024 -d N 256 1024 -d K 1024

    # Quick square sizes (shortcut)
    python -m triton_runner.bench.cross_version -k kernels/matmul.py \\
        --square 256 512 1024 2048 4096
"""

import argparse
import json
import sys
from pathlib import Path

from .problem_space import ProblemSpace
from .runner import CrossVersionRunner


def _parse_dimensions(args: list[str]) -> dict[str, list]:
    """Parse ``-d M 256 512 1024 -d N 256 512`` → {M: [256,512,1024], N: [256,512]}."""
    dims: dict[str, list] = {}
    i = 0
    while i < len(args):
        name = args[i]
        i += 1
        vals = []
        while i < len(args) and not args[i].startswith("-"):
            v = args[i]
            try:
                vals.append(int(v))
            except ValueError:
                try:
                    vals.append(float(v))
                except ValueError:
                    vals.append(v)
            i += 1
        dims[name] = vals
    return dims


def _parse_fixed(args: list[str]) -> dict:
    """Parse ``--fixed N=1024 K=1024`` → {N: 1024, K: 1024}."""
    fixed: dict = {}
    for item in args:
        k, v = item.split("=", 1)
        try:
            fixed[k] = int(v)
        except ValueError:
            try:
                fixed[k] = float(v)
            except ValueError:
                fixed[k] = v
    return fixed


def main(argv: list[str] | None = None):
    parser = argparse.ArgumentParser(
        description="Cross-version Triton kernel benchmark",
    )
    # kernel & envs
    parser.add_argument("-k", "--kernel", required=True,
                        help="Path to kernel .py file")
    parser.add_argument("-o", "--output-dir", default="cross_version_results",
                        help="Output directory (default: cross_version_results/)")
    parser.add_argument("-e", "--envs", nargs="+", default=[],
                        help="Conda env names (auto-discover if omitted)")
    parser.add_argument("-p", "--env-prefix", default="triton-",
                        help="Prefix for auto-discovering conda envs (default: triton-)")

    # problem space — one of: --sizes-json, --sweep, --product, --square, or auto-detect
    ps_group = parser.add_argument_group("problem space (choose one)")
    ps_group.add_argument("--sizes-json", default="",
                          help="Problem sizes JSON file or kernel .py with get_problem_space()")
    ps_group.add_argument("--sweep", default="",
                          help="Dimension to sweep (e.g. 'M'), use with -v/--values and --fixed")
    ps_group.add_argument("-v", "--values", nargs="+", type=int, default=[],
                          help="Values for the sweep dimension")
    ps_group.add_argument("--fixed", nargs="+", default=[],
                          help="Fixed values for non-sweep dims, e.g. N=1024 K=1024")
    ps_group.add_argument("--product", action="store_true",
                          help="Cartesian product mode, use with -d/--dim")
    ps_group.add_argument("-d", "--dim", nargs="+", action="append", default=[],
                          help="Dimension values, e.g. -d M 256 512 1024 (repeatable)")
    ps_group.add_argument("--square", nargs="+", type=int, default=[],
                          help="Square sizes shortcut: M=N=K for each value")

    # benchmark params
    parser.add_argument("--warmup", type=int, default=25,
                        help="Warmup time in ms (default: 25)")
    parser.add_argument("--rep", type=int, default=100,
                        help="Repetition time in ms (default: 100)")
    parser.add_argument("--lock-sm", type=int, default=0,
                        help="Lock GPU SM clock (MHz)")
    parser.add_argument("--lock-mem", type=int, default=0,
                        help="Lock GPU memory clock (MHz)")

    # output
    parser.add_argument("--csv", action="store_true", help="Generate CSV report")
    parser.add_argument("--md", action="store_true", help="Generate Markdown report")
    parser.add_argument("--no-table", action="store_true",
                        help="Suppress terminal table output")
    parser.add_argument("--capture-inputs", action="store_true",
                        help="Generate + save inputs once before benchmarking (ensures identical data across versions)")
    parser.add_argument("--save-cubin", action="store_true",
                        help="Save compiled cubin from each Triton version to output_dir/cubins/")
    parser.add_argument("--setup-only", action="store_true",
                        help="Only setup envs, skip benchmark")

    args = parser.parse_args(argv)

    # ---- resolve problem space ----
    sizes: list[dict] | ProblemSpace | str | None = None

    if args.square:
        sizes = ProblemSpace.matmul_square(args.square)
    elif args.product and args.dim:
        all_dims: dict[str, list] = {}
        for entry in args.dim:
            all_dims.update(_parse_dimensions(entry))
        sizes = ProblemSpace(dimensions=all_dims, mode="product")
    elif args.sweep and args.values:
        fixed = _parse_fixed(args.fixed) if args.fixed else {}
        sizes = ProblemSpace(
            dimensions={args.sweep: args.values},
            mode="sweep",
            sweep_dim=args.sweep,
            fixed=fixed,
        )
    elif args.sizes_json:
        sizes = args.sizes_json  # pass as string, runner resolves via ProblemSpace.from_json / from_kernel
    # else: None → auto-detect from kernel module

    # lock clocks
    lock_clocks = None
    if args.lock_sm and args.lock_mem:
        lock_clocks = (args.lock_sm, args.lock_mem)

    runner = CrossVersionRunner(
        kernel=args.kernel,
        output_dir=args.output_dir,
        envs=args.envs if args.envs else None,
        env_prefix=args.env_prefix,
        sizes=sizes,
        warmup=args.warmup,
        rep=args.rep,
        lock_clocks=lock_clocks,
        save_cubin=args.save_cubin,
    )

    if args.setup_only:
        runner.setup_envs()
        return

    runner.setup_envs()

    if args.capture_inputs:
        runner.capture_inputs()
    elif not runner.has_captured_inputs:
        print("[note] Inputs not captured — each env generates fresh random data. "
              "Use --capture-inputs for identical inputs across versions.",
              file=sys.stderr)

    runner.run()
    runner.compare(
        output_csv="report.csv" if args.csv else "",
        output_md="report.md" if args.md else "",
        print_table=not args.no_table,
    )


if __name__ == "__main__":
    main()
