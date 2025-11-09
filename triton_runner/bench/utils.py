# This file is adapted from:
# https://github.com/pytorch-labs/tritonbench/blob/main/tritonbench/utils/triton_op.py
#
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch
import time
import os
from triton_runner.testing import do_bench


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
                os.environ["RUNNER_PROD"] = "1"
                input_iter = list(self.get_input_iter())
                # sum_time = 0
                input_len = len(input_iter)
                for idx, input_args in enumerate(input_iter):
                    fn = method(self, *input_args)
                    elapsed_time = do_bench(fn)
                    elapsed_time_str = f"{elapsed_time:8.3f} ms"
                    if unit_name == "us":
                        elapsed_time_str = f"{elapsed_time * 1e3:8.3f} us"
                    if idx == input_len - 1:
                        print(f"[{name:<50}|] time: {elapsed_time_str}")
                    # sum_time += elapsed_time
                # print(f"[{name + " average":<30}|] time: {sum_time/input_len:.6f} ms")
                os.environ.pop("RUNNER_PROD", None)
            else:
                return method(self, *args[0])

        return wrapper

    return decorator
