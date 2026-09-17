"""
Persistent Kernels
==================

So far, we have defined kernels such that one programs handles one block of work
and we span all the work using the grid dimensions. This creates a large number
of programs, and we rely on the GPU to schedule the work. The primary benefit is
the GPU will dynamically load-balance the work across its SMs.

However, this approach has downsides. The scheduler incurs an overhead, and the
GPU is not aware of the memory access patterns of the kernels. This also
prevents overlapping across blocks of work, as the GPU waits until kernels have
fully exited before issuing more work.

Persistent kernels is a technique where we assign multiple blocks of work to
each program, and the programs "persist" on the GPU until all the work is
complete. The work assignment is typically static, although dynamic scheduling
is still possible with more advanced techniques or hardware features like
cluster launch control.

In this tutorial, we will explore persistent kernels by implementing a
persistent matmul. We will then show how we can pipeline across the persistent
outer loop to achieve greater overlap and more throughput.

Porting note (sm120): consumer Blackwell (sm120) has neither Hopper WGMMA nor
datacenter-Blackwell tcgen05/TMEM tensor cores. The MMA abstraction below keeps
the tcgen05 wrapper (MMAv5, sm100) and adds an Ampere ``mma_v2`` wrapper
(``mma.sync``) so the same kernels run on sm120 GPUs such as the RTX 5090.
The Hopper WGMMA wrapper of the upstream tutorial is dropped because it depends
on the Hopper-only tutorial ``05-wgmma.py``, which is not part of this port.
"""

import itertools
import pytest
import torch
import triton
import sys
from functools import partial
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from triton.experimental.gluon.nvidia.hopper import TensorDescriptor
from triton.experimental.gluon.language.nvidia.ampere import (
    NVMMADistributedLayout,
    DotOperandLayout,
    mma_v2,
)
from triton.experimental.gluon.language.nvidia.hopper import (
    tma,
    mbarrier,
    fence_async_shared,
)
from triton.experimental.gluon.language.nvidia.blackwell import (
    TensorMemoryLayout,
    tensor_memory_descriptor,
    allocate_tensor_memory,
    tcgen05_mma,
    tcgen05_commit,
)

if torch.cuda.is_available():
    from triton._C.libtriton import nvidia
    cublas_workspace = torch.empty(32 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    cublas = nvidia.cublas.CublasLt(cublas_workspace)
else:
    cublas = None


def is_hopper_or_newer():
    target = triton.runtime.driver.active.get_current_target()
    return target.backend == "cuda" and torch.cuda.get_device_capability()[0] >= 9


if __name__ == "__main__" and not is_hopper_or_newer():
    raise RuntimeError("This tutorial requires Hopper or newer NVIDIA GPU")

profiling_with_ncu = len(sys.argv) > 1 and sys.argv[1] == "profile"


def get_flops(ms, M, N, K):
    flops = 2 * M * N * K
    return flops * 1e-12 / (ms * 1e-3)


# %%
# In the previous tutorials of the upstream series, tensor core operations were
# introduced for Hopper (WGMMA) and datacenter Blackwell (tcgen05) GPUs. To make
# this tutorial more accessible, and to demonstrate some Gluon features, we build
# an abstraction around the different sets of tensor core operations so that our
# persistent matmul can be used across architectures.
#
# We can use @gluon.aggregate to define a class that contains the state of the
# matmul. We define the API of our MMA wrapper to be like WGMMA's, because it is
# the more restrictive of the two.


# MMA wrapper for Ampere-style mma.sync (MMAv2). This is the only tensor-core
# instruction available on sm120, which has neither WGMMA nor tcgen05. Unlike
# WGMMA/tcgen05, mma.sync consumes register operands and is synchronous: the
# wrapper loads the shared-memory operand tiles directly into dot-operand
# layouts, and the wait operation is a no-op.
@gluon.aggregate
class MMAv2:
    acc: gl.tensor
    use_acc: gl.tensor
    acc_layout: gl.constexpr
    lhs_layout: gl.constexpr
    rhs_layout: gl.constexpr

    @gluon.jit
    def initialize(dtype: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, num_warps: gl.constexpr):
        acc_layout: gl.constexpr = NVMMADistributedLayout(version=[2, 0], warps_per_cta=[num_warps, 1],
                                                          instr_shape=[16, 8])
        k_width: gl.constexpr = max(32 // dtype.primitive_bitwidth, 1)
        lhs_layout: gl.constexpr = DotOperandLayout(parent=acc_layout, operand_index=0, k_width=k_width)
        rhs_layout: gl.constexpr = DotOperandLayout(parent=acc_layout, operand_index=1, k_width=k_width)
        acc = gl.zeros((BLOCK_M, BLOCK_N), dtype=gl.float32, layout=acc_layout)
        return MMAv2(acc, gl.to_tensor(False), acc_layout, lhs_layout, rhs_layout)

    @gluon.jit
    def issue_async_mma(self, a, b):
        # Load the shared-memory tiles straight into the dot-operand layouts.
        # The accumulator starts from zeros, so plain accumulation implements
        # both the use_acc=True and use_acc=False cases.
        a_t = a.load(self.lhs_layout)
        b_t = b.load(self.rhs_layout)
        acc = mma_v2(a_t, b_t, self.acc)
        # Note that aggregates don't support in-place mutation, so we need to
        # return a new instance and re-assign it at the callsite.
        return MMAv2(acc, gl.to_tensor(True), self.acc_layout, self.lhs_layout, self.rhs_layout)

    @gluon.jit
    def wait_num_outstanding(self, num_outstanding: gl.constexpr):
        # mma.sync is synchronous.
        return self

    # Take the result and reset the accumulator.
    @gluon.jit
    def take_result(self, splitn: gl.constexpr = False):
        acc = gl.zeros(self.acc.shape, gl.float32, self.acc_layout)
        return self.acc, MMAv2(acc, gl.to_tensor(False), self.acc_layout, self.lhs_layout, self.rhs_layout)


# MMA wrapper for tcgen05. In order to implement `wait_num_outstanding`, we
# need to allocate barriers and keep track of how many MMAs have been issued.
# State will be tracked with an accumulator.
@gluon.aggregate
class MMAv5:
    use_acc: gl.tensor
    acc_tmem: tensor_memory_descriptor
    bar: gl.shared_memory_descriptor
    counter: gl.tensor

    @gluon.jit
    def initialize(dtype: gl.constexpr, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr, num_warps: gl.constexpr):
        layout: gl.constexpr = TensorMemoryLayout([BLOCK_M, BLOCK_N], col_stride=1)
        acc_tmem = allocate_tensor_memory(gl.float32, [BLOCK_M, BLOCK_N], layout)
        bar = gl.allocate_shared_memory(gl.int64, [1], mbarrier.MBarrierLayout())
        mbarrier.init(bar, count=1)
        return MMAv5(gl.to_tensor(False), acc_tmem, bar, gl.to_tensor(0))

    @gluon.jit
    def issue_async_mma(self, a, b):
        tcgen05_mma(a, b, self.acc_tmem, use_acc=self.use_acc)
        tcgen05_commit(self.bar)
        return MMAv5(gl.to_tensor(True), self.acc_tmem, self.bar, self.counter + 1)

    @gluon.jit
    def wait_num_outstanding(self, num_outstanding: gl.constexpr):
        mbarrier.wait(self.bar, (self.counter - 1 - num_outstanding) & 1)
        return self

    @gluon.jit
    def take_result(self, splitn: gl.constexpr = False):
        next = MMAv5(gl.to_tensor(False), self.acc_tmem, self.bar, self.counter)
        if splitn:
            layout: gl.constexpr = self.acc_tmem.get_reg_layout(instr_variant="32x32b_splitn")
            return self.acc_tmem.load(layout), next
        else:
            return self.acc_tmem.load(), next


def select_mma_impl():
    if torch.cuda.get_device_capability()[0] == 10:
        return MMAv5
    elif torch.cuda.get_device_capability()[0] in (8, 12):
        # sm120 (consumer Blackwell) and Ampere use mma.sync.
        return MMAv2
    else:
        return None


# %%
# Let's validate our abstraction by implementing a matmul where we pipeline both
# the MMA and the loads. This achieves async overlap of both the TMA loads and
# the MMAs by requiring at least two operand buffers. This will make the
# persistent kernel more interesting by allowing us to overlap more things.
#
# We will factor our kernel into components we can re-use between
# implementations.


@gluon.jit
def issue_loads(producer, a_desc, b_desc, off_m, off_n, k, bars, a_bufs, b_bufs, num_buffers: gl.constexpr, pred=True):
    index = producer % num_buffers
    producer += 1
    bar = bars.index(index)
    mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes, pred=pred)
    tma.async_load(a_desc, [off_m, k], bar, a_bufs.index(index), pred)
    tma.async_load(b_desc, [k, off_n], bar, b_bufs.index(index), pred)
    return producer


@gluon.jit
def issue_mma(consumer, mma, bars, a_bufs, b_bufs, num_buffers: gl.constexpr):
    index = consumer % num_buffers
    phase = consumer // num_buffers & 1
    consumer += 1
    mbarrier.wait(bars.index(index), phase)
    mma = mma.wait_num_outstanding(0)
    mma = mma.issue_async_mma(a_bufs.index(index), b_bufs.index(index))
    return consumer, mma


@gluon.jit
def matmul_pipelined_kernel(a_desc, b_desc, c_desc, MMAImpl: gl.constexpr, num_buffers: gl.constexpr,
                            num_warps: gl.constexpr):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype
    K = a_desc.shape[1]

    gl.static_assert(num_buffers >= 2, "expected at least 2 buffers")
    a_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + a_desc.block_type.shape, a_desc.layout)
    b_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + b_desc.block_type.shape, b_desc.layout)
    bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(num_buffers):
        mbarrier.init(bars.index(i), count=1)
    # Separate producer and consumer indices, to support more than 2 buffers.
    producer = 0
    consumer = 0

    pid_m = gl.program_id(axis=0)
    pid_n = gl.program_id(axis=1)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N

    # Use our MMA abstraction!
    mma = MMAImpl.initialize(dtype, BLOCK_M, BLOCK_N, num_warps)

    # Prefetch at most num_buffers-2 loads to allow the MMA to overlap.
    for k in gl.static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):
        producer = issue_loads(producer, a_desc, b_desc, off_m, off_n, k, bars, a_bufs, b_bufs, num_buffers)

    for k in range(BLOCK_K * (num_buffers - 2), K, BLOCK_K):
        producer = issue_loads(producer, a_desc, b_desc, off_m, off_n, k, bars, a_bufs, b_bufs, num_buffers)
        consumer, mma = issue_mma(consumer, mma, bars, a_bufs, b_bufs, num_buffers)

    for _ in gl.static_range(num_buffers - 2):
        consumer, mma = issue_mma(consumer, mma, bars, a_bufs, b_bufs, num_buffers)

    mma = mma.wait_num_outstanding(0)
    c_smem = gl.allocate_shared_memory(dtype, c_desc.block_type.shape, c_desc.layout)
    c, mma = mma.take_result()
    c_smem.store(c.to(dtype))
    fence_async_shared()
    tma.async_store(c_desc, [off_m, off_n], c_smem)
    tma.store_wait(pendings=0)


def matmul_pipelined(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps):
    MMAImpl = select_mma_impl()
    M, N = C.shape

    a_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float16)
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K], a_layout)
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_K, BLOCK_N], b_layout)
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], c_layout)

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_pipelined_kernel[grid](a_desc, b_desc, c_desc, MMAImpl, num_buffers, num_warps=num_warps)


@pytest.mark.parametrize("M, N, K", [(2000, 1000, 2000)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(128, 256, 64)])
@pytest.mark.parametrize("num_buffers", [2, 3, 4])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires Hopper or newer")
def test_pipelined_matmul(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps):
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)
    matmul_pipelined(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps)
    torch.testing.assert_close(A @ B, C, rtol=1e-3, atol=1e-1)


# %%
# The optimal block shapes for our kernel are BLOCK_M=128 and BLOCK_N=256, which
# gives the maximum instruction shape on both Blackwell and Hopper. However, on
# Hopper we need 8 warps to fit the accumulator in registers.

if __name__ == "__main__":
    M, N, K = 8192, 8192, 16 * 1024
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)

if __name__ == "__main__" and not profiling_with_ncu:
    BLOCK_M = 128
    BLOCK_N = 256
    is_hopper = torch.cuda.get_device_capability()[0] == 9
    warps = [8] if is_hopper else [4, 8]
    print("Benchmarking pipelined matmul")
    print("=============================")
    print(f"BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}")
    print("BLOCK_K num_buffers num_warps tflops/s")
    for (BLOCK_K, num_buffers), num_warps in itertools.product([(128, 2), (64, 3), (64, 4)], warps):
        print(f"{BLOCK_K:>7} {num_buffers:>11} {num_warps:>9}", end=" ")
        fn = lambda: matmul_pipelined(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps)
        ms = triton.testing.do_bench_cudagraph(fn)
        print(f"{get_flops(ms, M, N, K):8.2f}")
    print()

# %%
# To make the kernel persistent, all we have to do is put an outer loop around
# the kernel and iterate over the output tiles assigned to that kernel.
#
# Let's define a tile scheduler abstraction that will allow us to change the
# scheduling strategy, starting with a basic row-major tile scheduler.


@gluon.aggregate
class PersistentTileScheduler:
    pid_start: gl.tensor
    pid_end: gl.tensor
    num_pid_m: gl.tensor

    @gluon.jit
    def initialize(M, N, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr):
        kernel_id = gl.program_id(axis=0)
        num_kernels = gl.num_programs(axis=0)
        num_pid_m = gl.cdiv(M, BLOCK_M)
        num_pid_n = gl.cdiv(N, BLOCK_N)
        num_pid = num_pid_m * num_pid_n
        pid_per_kernel = gl.cdiv(num_pid, num_kernels)
        pid_start = kernel_id * pid_per_kernel
        pid_end = min(pid_start + pid_per_kernel, num_pid)
        return PersistentTileScheduler(pid_start, pid_end, num_pid_m)

    @gluon.jit
    def get_num_tiles(self):
        return self.pid_end - self.pid_start

    @gluon.jit
    def get_tile(self, idx):
        # Delinearize the tile ID along M.
        pid = self.pid_start + idx
        pid_m = pid % self.num_pid_m
        pid_n = pid // self.num_pid_m
        return pid_m, pid_n


# %%
# We can make the kernel persistent by literally placing the outer loop around
# the whole kernel, but let's re-use the TMA barrier and MMA state.
# We must scope the operand buffers to the inner loop so the shared memory
# allocator knows their liveranges do not intersect with the TMA store buffer.


@gluon.jit
def persistent_matmul_kernel(a_desc, b_desc, c_desc, MMAImpl: gl.constexpr, SchedulerImpl: gl.constexpr,
                             num_buffers: gl.constexpr, num_warps: gl.constexpr):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype
    K = a_desc.shape[1]

    bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(num_buffers):
        mbarrier.init(bars.index(i), count=1)
    # Producer and consumer indices.
    producer = 0
    consumer = 0

    mma = MMAImpl.initialize(dtype, BLOCK_M, BLOCK_N, num_warps)
    scheduler = SchedulerImpl.initialize(c_desc.shape[0], c_desc.shape[1], BLOCK_M, BLOCK_N)
    for idx in range(scheduler.get_num_tiles()):
        pid_m, pid_n = scheduler.get_tile(idx)
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N

        a_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + a_desc.block_type.shape, a_desc.layout)
        b_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + b_desc.block_type.shape, b_desc.layout)
        for k in gl.static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):
            producer = issue_loads(producer, a_desc, b_desc, off_m, off_n, k, bars, a_bufs, b_bufs, num_buffers)

        for k in range(BLOCK_K * (num_buffers - 2), K, BLOCK_K):
            producer = issue_loads(producer, a_desc, b_desc, off_m, off_n, k, bars, a_bufs, b_bufs, num_buffers)
            consumer, mma = issue_mma(consumer, mma, bars, a_bufs, b_bufs, num_buffers)

        for _ in gl.static_range(num_buffers - 2):
            consumer, mma = issue_mma(consumer, mma, bars, a_bufs, b_bufs, num_buffers)

        mma = mma.wait_num_outstanding(0)
        c_smem = gl.allocate_shared_memory(dtype, c_desc.block_type.shape, c_desc.layout)
        c, mma = mma.take_result()
        c_smem.store(c.to(dtype))
        fence_async_shared()
        tma.async_store(c_desc, [off_m, off_n], c_smem)
        tma.store_wait(pendings=0)


def persistent_matmul(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps, SchedulerImpl):
    M, N = C.shape
    MMAImpl = select_mma_impl()

    a_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float16)
    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K], a_layout)
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_K, BLOCK_N], b_layout)
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], c_layout)

    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    num_pid = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (min(num_sms, num_pid), )
    persistent_matmul_kernel[grid](a_desc, b_desc, c_desc, MMAImpl, SchedulerImpl, num_buffers, num_warps=num_warps)


schedulers = [PersistentTileScheduler]


@pytest.mark.parametrize("M, N, K", [(2000, 1000, 2000)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(128, 256, 64)])
@pytest.mark.parametrize("num_buffers", [2, 3, 4])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.parametrize("SchedulerImpl", schedulers)
@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires Hopper or newer")
def test_persistent_matmul(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps, SchedulerImpl):
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)
    persistent_matmul(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps, SchedulerImpl)
    torch.testing.assert_close(A @ B, C, rtol=1e-3, atol=1e-1)


if __name__ == "__main__" and not profiling_with_ncu:
    print("Benchmarking persistent matmul")
    print("==============================")
    print(f"BLOCK_M={BLOCK_M} BLOCK_N={BLOCK_N}")
    print("BLOCK_K num_buffers num_warps tflops/s")
    for (BLOCK_K, num_buffers), num_warps in itertools.product([(128, 2), (64, 3), (64, 4)], warps):
        print(f"{BLOCK_K:>7} {num_buffers:>11} {num_warps:>9}", end=" ")
        fn = lambda: persistent_matmul(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps,
                                       PersistentTileScheduler)
        ms = triton.testing.do_bench_cudagraph(fn)
        print(f"{get_flops(ms, M, N, K):8.2f}")
    print()

# %%
# The persistent kernel's L2 hit rate can be lower than the nonpersistent
# kernel's. We can improve L2 efficiency by "super-grouping" the tiles along
# columns. See 03-matrix-multiplication.py in the upstream Triton tutorials for
# more details. Let's encode this strategy in a new tile scheduler.


def GroupedPersistentTileScheduler(GROUP_SIZE_M):
    # Bind this as a constexpr so it can be captured.
    GROUP_SIZE_M = gl.constexpr(GROUP_SIZE_M)

    # Like C++ templates!
    @gluon.aggregate
    class GroupedPersistentTileSchedulerImpl:
        start_pid: gl.tensor
        num_pid_m: gl.tensor
        num_pid_in_group: gl.tensor
        num_pid: gl.tensor

        @gluon.jit
        def initialize(M, N, BLOCK_M: gl.constexpr, BLOCK_N: gl.constexpr):
            start_pid = gl.program_id(axis=0)
            num_pid_m = gl.cdiv(M, BLOCK_M)
            num_pid_n = gl.cdiv(N, BLOCK_N)
            num_pid_in_group = GROUP_SIZE_M * num_pid_n
            num_pid = num_pid_m * num_pid_n
            return GroupedPersistentTileSchedulerImpl(start_pid, num_pid_m, num_pid_in_group, num_pid)

        @gluon.jit
        def get_num_tiles(self):
            return gl.cdiv(self.num_pid - self.start_pid, gl.num_programs(axis=0))

        @gluon.jit
        def get_tile(self, idx):
            tile_id = self.start_pid + idx * gl.num_programs(axis=0)
            group_id = tile_id // self.num_pid_in_group
            first_pid_m = group_id * GROUP_SIZE_M
            group_size_m = min(self.num_pid_m - first_pid_m, GROUP_SIZE_M)
            pid_m = first_pid_m + (tile_id % group_size_m)
            pid_n = (tile_id % self.num_pid_in_group) // group_size_m
            return pid_m, pid_n

    GroupedPersistentTileSchedulerImpl.__name__ = f"GroupedPersistentTileScheduler({GROUP_SIZE_M.value})"
    return GroupedPersistentTileSchedulerImpl


# Add this to the testsuite.
schedulers += [GroupedPersistentTileScheduler(1), GroupedPersistentTileScheduler(8)]

if __name__ == "__main__" and not profiling_with_ncu:
    num_warps = 8 if is_hopper else 4
    num_buffers = 3 if is_hopper else 4
    print("Benchmarking grouped scheduler")
    print("=============================")
    print(f"BLOCK_M={BLOCK_M} BLOCK_N={BLOCK_N} BLOCK_K={BLOCK_K}")
    print(f"num_buffers={num_buffers} num_warps={num_warps}")
    print("GROUP_SIZE_M tflops/s")
    for GROUP_SIZE_M in [1, 2, 4, 6, 8]:
        print(f"{GROUP_SIZE_M:>12}", end=" ")
        fn = lambda: persistent_matmul(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps,
                                       GroupedPersistentTileScheduler(GROUP_SIZE_M))
        ms = triton.testing.do_bench_cudagraph(fn)
        print(f"{get_flops(ms, M, N, K):8.2f}")
    print()

# %%
# Pipelining across the outer loop benefits smaller K shapes more because a
# larger proportion of time is spent in the epilogue. We can try overlapping the
# TMA store with the next tile by rotating the TMA store wait.
#
# However, this causes the liverange of the TMA store buffer to overlap with the
# operand buffers, decreasing our max num_buffers to 3. There are 3 remedies:
#
# 1. Use gl.store which does not require shared memory but it cannot be
#    pipelined. However, the layout conversion requires shared memory.
# 2. Break up the TMA store to multiple steps, allowing us to use smaller
#    buffers, we will only be able to pipeline the last step.
#    reduces the amount of overlap.
# 3. Borrow one of the b_bufs.
#
# For BLOCK_{M,N,K} = (128, 256, 64), one B buffer is half the size of the
# accumulator, but we have enough memory to use 5 buffers for B just so that we
# can steal two buffers for the epilogue, even though the inner loop only uses
# 4 at a time.


# Forked versions of issue_loads and issue_mma that support `stealb`.
@gluon.jit
def issue_loads_stealb(producer, a_desc, b_desc, off_m, off_n, k, bars, a_bufs, b_bufs, stealb: gl.constexpr,
                       num_buffers: gl.constexpr, pred=True):
    index = producer % num_buffers
    b_index = producer % (num_buffers + stealb)
    producer += 1
    bar = bars.index(index)
    mbarrier.expect(bar, a_desc.block_type.nbytes + b_desc.block_type.nbytes, pred=pred)
    tma.async_load(a_desc, [off_m, k], bar, a_bufs.index(index), pred)
    tma.async_load(b_desc, [k, off_n], bar, b_bufs.index(b_index), pred)
    return producer


@gluon.jit
def issue_mma_stealb(consumer, mma, bars, a_bufs, b_bufs, stealb: gl.constexpr, num_buffers: gl.constexpr):
    index = consumer % num_buffers
    b_index = consumer % (num_buffers + stealb)
    phase = consumer // num_buffers & 1
    consumer += 1
    mbarrier.wait(bars.index(index), phase)
    mma = mma.wait_num_outstanding(0)
    mma = mma.issue_async_mma(a_bufs.index(index), b_bufs.index(b_index))
    return consumer, mma


@gluon.jit
def persistent_matmul_pipelined_kernel(a_desc, b_desc, c_desc, c_half_desc, MMAImpl: gl.constexpr,
                                       SchedulerImpl: gl.constexpr, num_buffers: gl.constexpr, STEALB: gl.constexpr,
                                       num_warps: gl.constexpr):
    BLOCK_M: gl.constexpr = c_desc.block_type.shape[0]
    BLOCK_N: gl.constexpr = c_desc.block_type.shape[1]
    BLOCK_K: gl.constexpr = a_desc.block_type.shape[1]
    dtype: gl.constexpr = a_desc.dtype
    K = a_desc.shape[1]

    # All buffers share the same liverange.
    gl.static_assert(num_buffers >= 3, "expected at least 3 buffers")
    a_bufs = gl.allocate_shared_memory(dtype, [num_buffers] + a_desc.block_type.shape, a_desc.layout)
    # Add an extra B buffer when stealing.
    b_bufs = gl.allocate_shared_memory(dtype, [num_buffers + STEALB] + b_desc.block_type.shape, b_desc.layout)
    if not STEALB:
        c_smem = gl.allocate_shared_memory(dtype, c_desc.block_type.shape, c_desc.layout)
    else:
        gl.static_assert(BLOCK_M == BLOCK_K or BLOCK_M == 2 * BLOCK_K,
                         "expected one or two B tiles to cover the epilogue tile")
        if BLOCK_M == 2 * BLOCK_K:
            # Upstream steals two B buffers and reinterprets them as half-tile C
            # buffers, but Triton 3.8.0 has no shared-memory reinterpret. Instead
            # we break the epilogue store into two half-tile stores from a single
            # half-sized C buffer (remedy 2 from the tutorial notes), which uses
            # the same amount of shared memory.
            c_smem = gl.allocate_shared_memory(dtype, c_half_desc.block_type.shape, c_half_desc.layout)
    bars = gl.allocate_shared_memory(gl.int64, [num_buffers, 1], mbarrier.MBarrierLayout())
    for i in gl.static_range(num_buffers):
        mbarrier.init(bars.index(i), count=1)
    producer = 0
    consumer = 0

    mma = MMAImpl.initialize(dtype, BLOCK_M, BLOCK_N, num_warps)
    scheduler = SchedulerImpl.initialize(c_desc.shape[0], c_desc.shape[1], BLOCK_M, BLOCK_N)
    num_tiles = scheduler.get_num_tiles()

    # Peeled inner loop prologue.
    idx = 0
    pid_m, pid_n = scheduler.get_tile(idx)
    off_m = pid_m * BLOCK_M
    off_n = pid_n * BLOCK_N
    for ki in gl.static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):
        producer = issue_loads_stealb(producer, a_desc, b_desc, off_m, off_n, ki, bars, a_bufs, b_bufs, STEALB,
                                      num_buffers)
    k = BLOCK_K * (num_buffers - 2)
    producer = issue_loads_stealb(producer, a_desc, b_desc, off_m, off_n, k, bars, a_bufs, b_bufs, STEALB, num_buffers)

    for _ in range(num_tiles):
        consumer, mma = issue_mma_stealb(consumer, mma, bars, a_bufs, b_bufs, STEALB, num_buffers)
        if STEALB:
            # Wait for the epilogue before the first TMA load.
            tma.store_wait(pendings=0)
        for k in range(BLOCK_K * (num_buffers - 1), K, BLOCK_K):
            producer = issue_loads_stealb(producer, a_desc, b_desc, off_m, off_n, k, bars, a_bufs, b_bufs, STEALB,
                                          num_buffers)
            consumer, mma = issue_mma_stealb(consumer, mma, bars, a_bufs, b_bufs, STEALB, num_buffers)

        epilogue_off_m = off_m
        epilogue_off_n = off_n

        # Peel the next prologue and fuse it with the pipeline drain loop.
        idx += 1
        pid_m, pid_n = scheduler.get_tile(idx)
        off_m = pid_m * BLOCK_M
        off_n = pid_n * BLOCK_N
        # Predicate the peeled prologue instead of using a conditional.
        pred = idx < num_tiles
        for ki in gl.static_range(0, BLOCK_K * (num_buffers - 2), BLOCK_K):
            producer = issue_loads_stealb(producer, a_desc, b_desc, off_m, off_n, ki, bars, a_bufs, b_bufs, STEALB,
                                          num_buffers, pred)
            consumer, mma = issue_mma_stealb(consumer, mma, bars, a_bufs, b_bufs, STEALB, num_buffers)
        k = BLOCK_K * (num_buffers - 2)
        producer = issue_loads_stealb(producer, a_desc, b_desc, off_m, off_n, k, bars, a_bufs, b_bufs, STEALB,
                                      num_buffers)

        mma = mma.wait_num_outstanding(0)
        use_split_n_load: gl.constexpr = STEALB and BLOCK_M != BLOCK_K
        c, mma = mma.take_result(splitn=use_split_n_load)
        c = c.to(dtype)
        if not STEALB:
            c_buf = c_smem
            tma.store_wait(pendings=0)
            c_buf.store(c)
            fence_async_shared()
            tma.async_store(c_desc, [epilogue_off_m, epilogue_off_n], c_buf)
        elif BLOCK_M == BLOCK_K:
            c_buf = b_bufs.index(producer % (num_buffers + STEALB))
            c_buf.store(c)
            fence_async_shared()
            tma.async_store(c_desc, [epilogue_off_m, epilogue_off_n], c_buf)
        else:
            # Split the epilogue store into two half-tile TMA stores from the
            # single half-sized C buffer, waiting for the first store to retire
            # before reusing the buffer.
            c0, c1 = c.reshape((BLOCK_M, 2, BLOCK_N // 2)).permute(0, 2, 1).split()
            c_smem.store(c0)
            fence_async_shared()
            tma.async_store(c_half_desc, [epilogue_off_m, epilogue_off_n], c_smem)
            tma.store_wait(pendings=0)
            c_smem.store(c1)
            fence_async_shared()
            tma.async_store(c_half_desc, [epilogue_off_m, epilogue_off_n + BLOCK_N // 2], c_smem)
    tma.store_wait(pendings=0)


def persistent_matmul_pipelined(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps, SchedulerImpl):
    M, N = C.shape
    MMAImpl = select_mma_impl()

    a_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_K], gl.float16)
    b_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_K, BLOCK_N], gl.float16)
    c_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N], gl.float16)
    c_half_layout = gl.NVMMASharedLayout.get_default_for([BLOCK_M, BLOCK_N // 2], gl.float16)

    a_desc = TensorDescriptor.from_tensor(A, [BLOCK_M, BLOCK_K], a_layout)
    b_desc = TensorDescriptor.from_tensor(B, [BLOCK_K, BLOCK_N], b_layout)
    c_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N], c_layout)
    c_half_desc = TensorDescriptor.from_tensor(C, [BLOCK_M, BLOCK_N // 2], c_half_layout)

    num_sms = torch.cuda.get_device_properties("cuda").multi_processor_count
    num_pid = triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N)
    grid = (min(num_sms, num_pid), )
    stealb = num_buffers == 4
    persistent_matmul_pipelined_kernel[grid](a_desc, b_desc, c_desc, c_half_desc, MMAImpl, SchedulerImpl, num_buffers,
                                             STEALB=stealb, num_warps=num_warps)


@pytest.mark.parametrize("M, N, K", [(208, 416, 304), (2000, 1000, 2000)])
@pytest.mark.parametrize("BLOCK_M, BLOCK_N, BLOCK_K", [(64, 64, 64), (128, 256, 64)])
@pytest.mark.parametrize("num_buffers", [3, 4])
@pytest.mark.parametrize("num_warps", [4, 8])
@pytest.mark.parametrize("SchedulerImpl", schedulers)
@pytest.mark.skipif(not is_hopper_or_newer(), reason="Requires Hopper or newer")
def test_persistent_matmul_pipelined(M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps, SchedulerImpl):
    torch.manual_seed(0)
    A = torch.randn(M, K, device="cuda", dtype=torch.float16)
    B = torch.randn(K, N, device="cuda", dtype=torch.float16)
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)
    persistent_matmul_pipelined(A, B, C, BLOCK_M, BLOCK_N, BLOCK_K, num_buffers, num_warps, SchedulerImpl)
    torch.testing.assert_close(A @ B, C, rtol=1e-3, atol=1e-1)


if __name__ == "__main__":
    args = {
        "BLOCK_M": 128,
        "BLOCK_N": 256,
        "BLOCK_K": 64,
        "num_buffers": 3 if is_hopper else 4,
        "num_warps": 8 if is_hopper else 4,
    }
    scheduler = PersistentTileScheduler if is_hopper else GroupedPersistentTileScheduler(8)
    nonpersistent = partial(matmul_pipelined, **args)
    persistent = partial(persistent_matmul, **args, SchedulerImpl=scheduler)
    persistent_pipelined = partial(persistent_matmul_pipelined, **args, SchedulerImpl=scheduler)

    M, N = 8192, 8192
    C = torch.empty(M, N, device="cuda", dtype=torch.float16)
    print("Benchmarking pipelined persistent")
    print("=================================")
    print("    K     nonpersistent    persistent   pipelined    cublas")
    for K in [2**i for i in range(9, 15)]:
        as_flops = partial(get_flops, M=M, N=N, K=K)
        A = torch.randn(M, K, device="cuda", dtype=torch.float16)
        B = torch.randn(K, N, device="cuda", dtype=torch.float16)
        BT = B.T.contiguous()
        r0 = as_flops(triton.testing.do_bench_cudagraph(lambda: nonpersistent(A, B, C)))
        r1 = as_flops(triton.testing.do_bench_cudagraph(lambda: persistent(A, B, C)))
        r2 = as_flops(triton.testing.do_bench_cudagraph(lambda: persistent_pipelined(A, B, C)))
        r3 = as_flops(triton.testing.do_bench(lambda: cublas.matmul(A, BT, C)))
        print(f"{K:>5} {r0:>17.2f} {r1:>13.2f} {r2:>11.2f} {r3:>9.2f}")

# %%
# Main takeaways:
#
# - Persistent kernels replace GPU block scheduling with a (typically) static
#   schedule. This allows more resource and compute coordination/overlap between
#   blocks at the cost of losing dynamic scheduling.
# - Persistent kernels tend to benefit smaller problem sizes, but still deliver
#   benefits for large problem sizes.
