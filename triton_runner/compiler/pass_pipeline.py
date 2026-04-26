"""Pass-level pipeline management for Triton compilation.

Enables starting compilation from any individual pass within a stage,
rather than only from stage boundaries.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from triton._C.libtriton import ir, passes, nvidia


@dataclass
class Pass:
    """A single compilation pass within a stage."""

    name: str               # short name for start_pass parameter: "inliner"
    dump_fragment: str       # fragment matching dump filename: "Inliner"
    add_fn: Callable         # pass registration function: fn(pm, *args, **kwargs)
    args: tuple = ()
    kwargs: dict = field(default_factory=dict)


def _run_single_pass(mod, p: Pass, context=None):
    """Create a pass manager, register a single pass, and run it on the module."""
    pm = ir.pass_manager(context or mod.context)
    p.add_fn(pm, *p.args, **p.kwargs)
    pm.run(mod)
    return mod


def _build_ttir_passes(capability: int):
    """Build the ordered list of TTIR passes."""
    passes_list = [
        Pass("inliner", "Inliner", passes.common.add_inliner),
        Pass("rewrite_tensor_pointer", "TritonRewriteTensorPointer", passes.ttir.add_rewrite_tensor_pointer),
    ]
    if capability // 10 < 9:
        passes_list.append(
            Pass("rewrite_tensor_descriptor_to_pointer", "RewriteTensorDescriptorToPointer",
                 passes.ttir.add_rewrite_tensor_descriptor_to_pointer))
    passes_list += [
        Pass("canonicalizer", "Canonicalizer", passes.common.add_canonicalizer),
        Pass("combine_ops", "TritonCombineOps", passes.ttir.add_combine),
        Pass("reorder_broadcast", "TritonReorderBroadcast", passes.ttir.add_reorder_broadcast),
        Pass("cse", "CSE", passes.common.add_cse),
        Pass("symbol_dce", "SymbolDCE", passes.common.add_symbol_dce),
        Pass("loop_unroll", "TritonLoopUnroll", passes.ttir.add_loop_unroll),
    ]
    return passes_list


def _build_ttgir_passes(capability: int, num_stages: int, cluster_info, maxnreg: Optional[int]):
    """Build the ordered list of TTGIR passes."""
    # Helper for pre-pass module attributes
    pre_setup = [
        ("set_maxnreg", maxnreg),
        ("set_cluster_info", cluster_info),
    ]

    passes_list = []

    # TTIR -> TTGIR conversion
    passes_list.append(
        Pass("convert_ttgpuir", "ConvertTritonToTritonGPU",
             passes.ttir.add_convert_to_ttgpuir,
             args=(f"cuda:{capability}", None, 32, None)))
    # Note: num_warps and num_ctas will be resolved when building the pipeline with actual options
    # We store placeholder and patch after build

    passes_list.append(Pass("coalesce", "TritonGPUCoalesce", passes.ttgpuir.add_coalesce))

    if capability // 10 >= 8:
        passes_list.append(Pass("f32_dot_tc", "TritonGPUF32DotTC", passes.ttgpuir.add_f32_dot_tc))

    passes_list.append(
        Pass("plan_cta", "TritonGPUPlanCTAPass",
             nvidia.passes.ttnvgpuir.add_plan_cta,
             args=(cluster_info,)))

    passes_list.append(
        Pass("remove_layout_conversions", "TritonGPURemoveLayoutConversions",
             passes.ttgpuir.add_remove_layout_conversions))

    passes_list.append(
        Pass("optimize_thread_locality", "TritonGPUOptimizeThreadLocality",
             passes.ttgpuir.add_optimize_thread_locality))

    passes_list.append(
        Pass("accelerate_matmul", "TritonGPUAccelerateMatmul",
             passes.ttgpuir.add_accelerate_matmul))

    passes_list.append(
        Pass("remove_layout_conversions_2", "TritonGPURemoveLayoutConversions",
             passes.ttgpuir.add_remove_layout_conversions))

    passes_list.append(
        Pass("optimize_dot_operands", "TritonGPUOptimizeDotOperands",
             passes.ttgpuir.add_optimize_dot_operands,
             args=(capability >= 80,)))

    passes_list.append(
        Pass("optimize_descriptor_encoding", "TritonNvidiaGPUOptimizeDescriptorEncodingPass",
             nvidia.passes.ttnvgpuir.add_optimize_descriptor_encoding))

    passes_list.append(
        Pass("loop_aware_cse", "TritonLoopAwareCSE",
             passes.ttir.add_loop_aware_cse))

    if capability // 10 in [8, 9]:
        passes_list.extend([
            Pass("fuse_nested_loops", "TritonGPUFuseNestedLoops", passes.ttgpuir.add_fuse_nested_loops),
            Pass("canonicalizer_2", "Canonicalizer", passes.common.add_canonicalizer),
            Pass("triton_licm", "TritonLoopInvariantCodeMotion", passes.ttir.add_triton_licm),
            Pass("canonicalizer_3", "Canonicalizer", passes.common.add_canonicalizer),
            Pass("combine_tensor_select_and_if", "TritonGPUCombineTensorSelectAndIf",
                 passes.ttgpuir.add_combine_tensor_select_and_if),
            Pass("hopper_warpspec", "TritonGPUAutomaticWarpSpecialization",
                 nvidia.passes.hopper.add_hopper_warpspec,
                 args=(num_stages, True)),
            Pass("assign_latencies", "TritonGPUAssignLatencies", passes.ttgpuir.add_assign_latencies,
                 args=(num_stages,)),
            Pass("schedule_loops", "TritonGPUScheduleLoops", passes.ttgpuir.add_schedule_loops),
            Pass("pipeline", "TritonGPUPipeline", passes.ttgpuir.add_pipeline,
                 args=(num_stages, True)),
        ])
    elif capability // 10 >= 10:
        passes_list.extend([
            Pass("fuse_nested_loops", "TritonGPUFuseNestedLoops", passes.ttgpuir.add_fuse_nested_loops),
            Pass("canonicalizer_2", "Canonicalizer", passes.common.add_canonicalizer),
            Pass("triton_licm", "TritonLoopInvariantCodeMotion", passes.ttir.add_triton_licm),
            Pass("optimize_accumulator_init", "TritonGPUOptimizeAccumulatorInit",
                 passes.ttgpuir.add_optimize_accumulator_init),
            Pass("hoist_tmem_alloc", "TritonGPUHoistTMEMAlloc", passes.ttgpuir.add_hoist_tmem_alloc),
            Pass("promote_lhs_to_tmem", "TritonNvidiaGPUPromoteLHSToTMemPass",
                 nvidia.passes.ttnvgpuir.add_promote_lhs_to_tmem),
            Pass("assign_latencies", "TritonGPUAssignLatencies", passes.ttgpuir.add_assign_latencies,
                 args=(num_stages,)),
            Pass("schedule_loops", "TritonGPUScheduleLoops", passes.ttgpuir.add_schedule_loops),
            Pass("warp_specialize", "TritonGPUAutomaticWarpSpecialization",
                 passes.ttgpuir.add_warp_specialize,
                 args=(num_stages,)),
            Pass("pipeline", "TritonGPUPipeline", passes.ttgpuir.add_pipeline,
                 args=(num_stages, True)),
            Pass("combine_tensor_select_and_if", "TritonGPUCombineTensorSelectAndIf",
                 passes.ttgpuir.add_combine_tensor_select_and_if),
            Pass("remove_tmem_tokens", "TritonNvidiaGPURemoveTMEMTokensPass",
                 nvidia.passes.ttnvgpuir.add_remove_tmem_tokens),
        ])
    else:
        passes_list.append(
            Pass("triton_licm", "TritonLoopInvariantCodeMotion", passes.ttir.add_triton_licm))

    # Common tail passes for all capabilities
    passes_list += [
        Pass("canonicalizer_tail", "Canonicalizer", passes.common.add_canonicalizer),
        Pass("loop_aware_cse_2", "TritonLoopAwareCSE", passes.ttir.add_loop_aware_cse),
        Pass("prefetch", "TritonGPUPrefetch", passes.ttgpuir.add_prefetch),
        Pass("optimize_dot_operands_2", "TritonGPUOptimizeDotOperands",
             passes.ttgpuir.add_optimize_dot_operands,
             args=(capability >= 80,)),
        Pass("coalesce_async_copy", "TritonGPUCoalesceAsyncCopy", passes.ttgpuir.add_coalesce_async_copy),
        Pass("optimize_tmem_layouts", "TritonNvidiaGPUOptimizeTMemLayoutsPass",
             nvidia.passes.ttnvgpuir.add_optimize_tmem_layouts),
        Pass("remove_layout_conversions_3", "TritonGPURemoveLayoutConversions",
             passes.ttgpuir.add_remove_layout_conversions),
        Pass("interleave_tmem", "TritonNvidiaGPUInterleaveTMemPass",
             nvidia.passes.ttnvgpuir.add_interleave_tmem),
        Pass("reduce_data_duplication", "TritonGPUReduceDataDuplication",
             passes.ttgpuir.add_reduce_data_duplication),
        Pass("reorder_instructions", "TritonGPUReorderInstructions",
             passes.ttgpuir.add_reorder_instructions),
        Pass("loop_aware_cse_3", "TritonLoopAwareCSE", passes.ttir.add_loop_aware_cse),
        Pass("symbol_dce_2", "SymbolDCE", passes.common.add_symbol_dce),
    ]

    if capability // 10 >= 9:
        passes_list += [
            Pass("tma_lowering", "TritonNvidiaGPUTMALoweringPass",
                 nvidia.passes.ttnvgpuir.add_tma_lowering),
            Pass("fence_insertion", "TritonGPUFenceInsertion",
                 nvidia.passes.ttnvgpuir.add_fence_insertion),
        ]

    passes_list += [
        Pass("sccp", "SCCP", passes.common.add_sccp),
        Pass("canonicalizer_final", "Canonicalizer", passes.common.add_canonicalizer),
    ]

    return passes_list, pre_setup


def _build_llir_passes(capability: int, ptx_version: int):
    """Build the ordered list of MLIR-level LLIR passes (before LLVM conversion)."""
    passes_list = [
        Pass("lower_mma", "TritonNvidiaGPUMMALoweringPass",
             nvidia.passes.ttnvgpuir.add_lower_mma),
        Pass("combine_tensor_select_and_if", "TritonGPUCombineTensorSelectAndIf",
             passes.ttgpuir.add_combine_tensor_select_and_if),
        Pass("allocate_warp_groups", "TritonGPUAllocateWarpGroups",
             passes.ttgpuir.add_allocate_warp_groups),
        Pass("scf_to_cf", "SCFToControlFlowPass",
             passes.convert.add_scf_to_cf),
        Pass("allocate_shared_memory", "AllocateSharedMemory",
             passes.ttgpuir.add_allocate_shared_memory),
        Pass("allocate_tensor_memory", "TritonTensorMemoryAllocationPass",
             nvidia.passes.ttnvgpuir.add_allocate_tensor_memory),
        Pass("allocate_global_scratch", "TritonGPUGlobalScratchAllocationPass",
             passes.ttgpuir.add_allocate_global_scratch_memory),
        Pass("to_llvmir", "ConvertTritonGPUToLLVM",
             nvidia.passes.ttgpuir.add_to_llvmir,
             args=(capability, ptx_version)),
        Pass("canonicalizer", "Canonicalizer", passes.common.add_canonicalizer),
        Pass("cse", "CSE", passes.common.add_cse),
        Pass("nvgpu_to_llvm", "ConvertNVGPUToLLVM",
             nvidia.passes.ttnvgpuir.add_nvgpu_to_llvm),
        Pass("warp_specialize_to_llvm", "ConvertWarpSpecializeToLLVM",
             nvidia.passes.ttnvgpuir.add_warp_specialize_to_llvm),
        Pass("canonicalizer_2", "Canonicalizer", passes.common.add_canonicalizer),
        Pass("cse_2", "CSE", passes.common.add_cse),
        Pass("symbol_dce", "SymbolDCE", passes.common.add_symbol_dce),
    ]
    return passes_list


class PassPipeline:
    """Manages the pass pipeline for a compilation stage."""

    def __init__(self, stage: str, passes_list: list[Pass]):
        self.stage = stage
        self.passes = passes_list

    def find_pass(self, name: str) -> int:
        """Return the index of a pass within this stage's pipeline.

        Raises ValueError if the pass name is not found.
        """
        for i, p in enumerate(self.passes):
            if p.name == name:
                return i
        raise ValueError(
            f"Pass '{name}' not found in {self.stage} pipeline. "
            f"Available: {[p.name for p in self.passes]}"
        )

    def run_from(self, mod, metadata, start_idx: int, context=None):
        """Run passes from start_idx (inclusive) to end."""
        for p in self.passes[start_idx:]:
            _run_single_pass(mod, p, context=context)
        return mod

    def run_all(self, mod, metadata, context=None):
        """Run all passes in this pipeline."""
        return self.run_from(mod, metadata, 0, context=context)

    def get_pass_names(self) -> list[str]:
        """Return all pass names in order."""
        return [p.name for p in self.passes]


def build_pipeline_for_stage(
    stage: str,
    capability: int,
    options=None,
    ptx_version: Optional[int] = None,
) -> PassPipeline:
    """Build a PassPipeline for the given compilation stage.

    Args:
        stage: One of "ttir", "ttgir", "llir"
        capability: SM capability (e.g., 80, 90, 100)
        options: Backend options object (CUDAOptions)
        ptx_version: PTX version (needed for LLIR)

    Returns:
        PassPipeline for the requested stage
    """
    if stage == "ttir":
        return PassPipeline("ttir", _build_ttir_passes(capability))

    if stage == "ttgir":
        num_stages = options.num_stages if options else 3
        cluster_info = nvidia.ClusterInfo()
        if options and options.cluster_dims is not None:
            cluster_info.clusterDimX = options.cluster_dims[0]
            cluster_info.clusterDimY = options.cluster_dims[1]
            cluster_info.clusterDimZ = options.cluster_dims[2]
        maxnreg = options.maxnreg if options else None
        passes_list, _ = _build_ttgir_passes(capability, num_stages, cluster_info, maxnreg)
        # Patch convert_ttgpuir with actual num_warps and num_ctas
        num_warps = options.num_warps if options else 4
        num_ctas = options.num_ctas if options else 1
        for p in passes_list:
            if p.name == "convert_ttgpuir":
                p.args = (f"cuda:{capability}", num_warps, 32, num_ctas)
                break
        return PassPipeline("ttgir", passes_list)

    if stage == "llir":
        if ptx_version is None:
            ptx_version = 86  # default for sm90
        return PassPipeline("llir", _build_llir_passes(capability, ptx_version))

    raise ValueError(f"Unknown stage: {stage}")


def run_stage_pre_setup(mod, metadata, options, capability, stage: str):
    """Run pre-pass setup for a stage (module attribute modifications).

    Returns any context that needs to be passed to subsequent stages.
    """
    if stage == "ttgir":
        if options and options.maxnreg is not None:
            mod.set_attr("ttg.maxnreg", ir.builder(mod.context).get_int32_attr(options.maxnreg))
    elif stage == "llir":
        pass  # No pre-setup for LLIR
    return mod
