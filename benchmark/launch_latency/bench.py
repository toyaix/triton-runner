# This file is adapted from:
# https://github.com/pytorch-labs/tritonbench/blob/main/tritonbench/tritonbench/operators/launch_latency/operator.py
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

from triton.compiler import CompiledKernel
from triton_runner.bench.launch_latency.kernels import get_trivial_add_kernel, nop_kernel, nop_with_args_kernel, runner_nop_kernel, runner_nop_with_args_kernel
from triton_runner.bench.utils import benchmark
from torch import zeros
import triton


class Operator:
    DEFAULT_METRICS = ["walltime"]

    def get_input_iter(self):
        yield tuple()
        targs = [zeros(1, device="cuda") for _ in range(5)]
        iargs = [1 for _ in range(9)]
        cargs = [32 for _ in range(5)]
        yield tuple([*targs, *iargs, *cargs])

    @benchmark("triton", "us")
    def nop_triton_kernel(self, *args):
        if len(args) == 0:
            return lambda: nop_kernel[
                1,
            ]()
        return lambda: nop_with_args_kernel[
            1,
        ](*args)

    @benchmark("triton_compiled", "us")
    def nop_triton_compiled_kernel_run(self, *args):
        if len(args) == 0:
            bin = nop_kernel[
                1,
            ]()

        else:
            bin = nop_with_args_kernel[
                1,
            ](*args)
            if triton.__version__ == "3.2.0":
                args = args[:-5]  # remove tl.constexpr args
        function = bin.function
        metadata = (bin.packed_metadata if hasattr(bin, "packed_metadata") else bin.metadata)
        if hasattr(CompiledKernel, "launch_metadata"):
            return lambda: bin.run(1, 1, 1, 0, function, metadata, None, None, None, *args)
        else:
            return lambda: bin.run(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, function, None, None, metadata, *args)

    @benchmark("triton_runner", "us")
    def nop_triton_runner_kernel(self, *args):
        if len(args) == 0:
            return lambda: nop_kernel[
                1,
            ]()
        return lambda: nop_with_args_kernel[
            1,
        ](*args)

    @benchmark("triton_runner_compiled", "us")
    def nop_triton_runner_kernel_run(self, *args):
        if len(args) == 0:
            bin = runner_nop_kernel[
                1,
            ]()

        else:
            bin = runner_nop_with_args_kernel[
                1,
            ](*args)
            if triton.__version__ == "3.2.0":
                args = args[:-5]  # remove tl.constexpr args
        function = bin.function
        metadata = (bin.packed_metadata if hasattr(bin, "packed_metadata") else bin.metadata)
        if hasattr(CompiledKernel, "launch_metadata"):
            return lambda: bin.run(1, 1, 1, 0, function, metadata, None, None, None, *args)
        else:
            return lambda: bin.run(1, 1, 1, 1, 1, 1, 1, 1, 0, 0, function, None, None, metadata, *args)

    @benchmark("inductor", "us")
    def nop_inductor_kernel(self, *args):
        trivial_add_kernel = get_trivial_add_kernel()
        return lambda: trivial_add_kernel(*args)

    @benchmark("python", "us")
    def nop_python_function(self, *args):

        def nop():
            pass

        return nop


if __name__ == "__main__":
    op = Operator()
    op.nop_python_function()
    op.nop_triton_kernel()
    op.nop_triton_compiled_kernel_run()
    op.nop_triton_runner_kernel()
    op.nop_triton_runner_kernel_run()
