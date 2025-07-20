# This file is adapted from:
# https://github.com/pytorch-labs/tritonbench/blob/main/tritonbench/utils/triton_op.py
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import time


class TimerContext:

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.elapsed_ms = None

    def __enter__(self):
        if self.enabled:
            self._start_time = time.perf_counter()
        return self

    def __exit__(self, *args, **kwargs):
        if self.enabled:
            end_time = time.perf_counter()
            self.elapsed_ms = (end_time - self._start_time) * 1e3


def benchmark(name, unit_name="ms"):

    def decorator(method):

        def wrapper(self, *args, **kwargs):
            if kwargs.pop("enable_benchmark", True) is not False:
                input_iter = list(self.get_input_iter())
                # sum_time = 0
                input_len = len(input_iter)
                for idx, input_args in enumerate(input_iter):
                    fn = method(self, *input_args)
                    elapsed_time = do_bench_walltime(fn)
                    elapsed_time_str = f"{elapsed_time:8.3f} ms"
                    if unit_name == "us":
                        elapsed_time_str = f"{elapsed_time * 1e3:8.3f} us"
                    if idx == input_len - 1:
                        print(f"[{name:<30}|] time: {elapsed_time_str}")
                    # sum_time += elapsed_time
                # print(f"[{name + " average":<30}|] time: {sum_time/input_len:.6f} ms")
            else:
                return method(self, *args[0])

        return wrapper

    return decorator


def do_bench_walltime(fn, warmup=25, rep=100):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    fn()
    torch.cuda.synchronize()

    with TimerContext() as timer:
        for _ in range(5):
            fn()
        torch.cuda.synchronize()
    estimate_ms = timer.elapsed_ms / 5

    # compute number of warmup and repeat
    n_warmup = max(1, int(warmup / estimate_ms))
    n_repeat = max(1, int(rep / estimate_ms))

    # Warm-up
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()

    # Benchmark
    start_time = time.perf_counter()
    for _ in range(n_repeat):
        fn()
    torch.cuda.synchronize()
    end_time = time.perf_counter()
    wall_time_ms = (end_time - start_time) * 1e3 / n_repeat
    return wall_time_ms
