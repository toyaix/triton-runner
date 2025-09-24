from triton.backends.compiler import Language
from triton._C.libtriton import ir, passes, llvm, nvidia
from triton.backends.nvidia.compiler import get_ptx_version_from_options, sm_arch_from_capability, get_features
from triton import knobs


def make_ttir(mod, metadata, opt, capability):
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()
    passes.common.add_inliner(pm)
    passes.ttir.add_rewrite_tensor_pointer(pm)
    if capability // 10 < 9:
        passes.ttir.add_rewrite_tensor_descriptor_to_pointer(pm)
    passes.common.add_canonicalizer(pm)
    passes.ttir.add_combine(pm)
    passes.ttir.add_reorder_broadcast(pm)
    passes.common.add_cse(pm)
    passes.common.add_symbol_dce(pm)
    passes.ttir.add_loop_unroll(pm)
    pm.run(mod)
    return mod


def make_ttgir(mod, metadata, opt, capability):
    # Set maxnreg on all kernels, if it was provided.
    if opt.maxnreg is not None:
        mod.set_attr("ttg.maxnreg", ir.builder(mod.context).get_int32_attr(opt.maxnreg))

    cluster_info = nvidia.ClusterInfo()
    if opt.cluster_dims is not None:
        cluster_info.clusterDimX = opt.cluster_dims[0]
        cluster_info.clusterDimY = opt.cluster_dims[1]
        cluster_info.clusterDimZ = opt.cluster_dims[2]
    pm = ir.pass_manager(mod.context)
    dump_enabled = pm.enable_debug()
    passes.ttir.add_convert_to_ttgpuir(pm, f"cuda:{capability}", opt.num_warps, 32, opt.num_ctas)
    # optimize TTGIR
    passes.ttgpuir.add_coalesce(pm)
    if capability // 10 >= 8:
        passes.ttgpuir.add_f32_dot_tc(pm)
    # TODO(Qingyi): Move PlanCTAPass to the front of CoalescePass
    nvidia.passes.ttnvgpuir.add_plan_cta(pm, cluster_info)
    passes.ttgpuir.add_remove_layout_conversions(pm)
    passes.ttgpuir.add_optimize_thread_locality(pm)
    passes.ttgpuir.add_accelerate_matmul(pm)
    passes.ttgpuir.add_remove_layout_conversions(pm)
    passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 80)
    nvidia.passes.ttnvgpuir.add_optimize_descriptor_encoding(pm)
    passes.ttir.add_loop_aware_cse(pm)
    if capability // 10 in [8, 9]:
        passes.ttgpuir.add_fuse_nested_loops(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_triton_licm(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttgpuir.add_combine_tensor_select_and_if(pm)
        nvidia.passes.hopper.add_hopper_warpspec(pm, opt.num_stages, dump_enabled)
        passes.ttgpuir.add_assign_latencies(pm, opt.num_stages)
        passes.ttgpuir.add_schedule_loops(pm)
        passes.ttgpuir.add_pipeline(pm, opt.num_stages, dump_enabled)
    elif capability // 10 >= 10:
        passes.ttgpuir.add_fuse_nested_loops(pm)
        passes.common.add_canonicalizer(pm)
        passes.ttir.add_triton_licm(pm)
        passes.ttgpuir.add_optimize_accumulator_init(pm)
        passes.ttgpuir.add_hoist_tmem_alloc(pm)
        nvidia.passes.ttnvgpuir.add_promote_lhs_to_tmem(pm)
        passes.ttgpuir.add_assign_latencies(pm, opt.num_stages)
        passes.ttgpuir.add_schedule_loops(pm)
        passes.ttgpuir.add_warp_specialize(pm, opt.num_stages)
        passes.ttgpuir.add_pipeline(pm, opt.num_stages, dump_enabled)
        passes.ttgpuir.add_combine_tensor_select_and_if(pm)
        nvidia.passes.ttnvgpuir.add_remove_tmem_tokens(pm)
    else:
        passes.ttir.add_triton_licm(pm)
    passes.common.add_canonicalizer(pm)
    passes.ttir.add_loop_aware_cse(pm)
    passes.ttgpuir.add_prefetch(pm)
    passes.ttgpuir.add_optimize_dot_operands(pm, capability >= 80)
    passes.ttgpuir.add_coalesce_async_copy(pm)
    nvidia.passes.ttnvgpuir.add_optimize_tmem_layouts(pm)
    passes.ttgpuir.add_remove_layout_conversions(pm)
    nvidia.passes.ttnvgpuir.add_interleave_tmem(pm)
    passes.ttgpuir.add_reduce_data_duplication(pm)
    passes.ttgpuir.add_reorder_instructions(pm)
    passes.ttir.add_loop_aware_cse(pm)
    passes.common.add_symbol_dce(pm)
    if capability // 10 >= 9:
        nvidia.passes.ttnvgpuir.add_tma_lowering(pm)
        nvidia.passes.ttnvgpuir.add_fence_insertion(pm)
    passes.common.add_sccp(pm)
    passes.common.add_canonicalizer(pm)
    pm.run(mod)
    metadata["cluster_dims"] = (cluster_info.clusterDimX, cluster_info.clusterDimY, cluster_info.clusterDimZ)
    tensordesc_meta = mod.get_tensordesc_metadata()
    metadata["tensordesc_meta"] = tensordesc_meta
    return mod

def ttgir_opt(src, metadata, options, capability):
    mod = src
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()

    passes.ttgpuir.add_inliner(pm)
    passes.common.add_sccp(pm)
    passes.ttir.add_loop_aware_cse(pm)
    passes.ttgpuir.add_canonicalizer(pm)
    passes.ttgpuir.add_combine_tensor_select_and_if(pm)

    pm.run(mod)
    metadata["tensordesc_meta"] = mod.get_tensordesc_metadata()
    return mod

def make_llir(src, metadata, options, capability, target_arch):
    ptx_version = get_ptx_version_from_options(options, target_arch)

    mod = src
    # TritonGPU -> LLVM-IR (MLIR)
    pm = ir.pass_manager(mod.context)
    pm.enable_debug()

    nvidia.passes.ttnvgpuir.add_lower_mma(pm)
    passes.ttgpuir.add_combine_tensor_select_and_if(pm)
    passes.ttgpuir.add_allocate_warp_groups(pm)
    passes.convert.add_scf_to_cf(pm)
    passes.ttgpuir.add_allocate_shared_memory(pm)
    nvidia.passes.ttnvgpuir.add_allocate_tensor_memory(pm)
    passes.ttgpuir.add_allocate_global_scratch_memory(pm)
    nvidia.passes.ttgpuir.add_to_llvmir(pm, capability, ptx_version)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    nvidia.passes.ttnvgpuir.add_nvgpu_to_llvm(pm)
    nvidia.passes.ttnvgpuir.add_warp_specialize_to_llvm(pm)
    passes.common.add_canonicalizer(pm)
    passes.common.add_cse(pm)
    passes.common.add_symbol_dce(pm)
    if not knobs.compilation.disable_line_info:
        passes.llvmir.add_di_scope(pm)
    pm.run(mod)
    # LLVM-IR (MLIR) -> LLVM-IR (LLVM)
    llvm.init_targets()
    context = llvm.context()
    if knobs.compilation.enable_asan:
        raise RuntimeError(
            "Address Sanitizer Error: Address sanitizer is currently only supported on the AMD backend")
    llvm_mod = llvm.to_module(mod, context)
    proc = sm_arch_from_capability(capability)
    features = get_features(options, target_arch)
    triple = 'nvptx64-nvidia-cuda'
    nvidia.set_short_ptr()
    llvm.attach_datalayout(llvm_mod, triple, proc, features)
    nvidia.set_nvvm_reflect_ftz(llvm_mod)

    if options.extern_libs:
        paths = [path for (name, path) in options.extern_libs]
        llvm.link_extern_libs(llvm_mod, paths)

    llvm.optimize_module(llvm_mod, llvm.OPTIMIZE_O3)

    # Get some metadata
    # warp-specialization mutates num_warps
    total_num_warps = src.get_int_attr("ttg.total-num-warps")
    if total_num_warps is not None:
        metadata["num_warps"] = total_num_warps
    metadata["shared"] = src.get_int_attr("ttg.shared")
    metadata["tmem_size"] = src.get_int_attr("ttg.tensor_memory_size")
    metadata["global_scratch_size"] = src.get_int_attr("ttg.global_scratch_memory_size")
    metadata["global_scratch_align"] = src.get_int_attr("ttg.global_scratch_memory_alignment")
    ret = str(llvm_mod)
    del llvm_mod
    del context
    return ret


def add_stages(backend, stages, options, language):
    capability = backend._parse_arch(options.arch)
    if language == Language.TRITON:
        stages["ttir"] = lambda src, metadata: make_ttir(src, metadata, options, capability)
        stages["ttgir"] = lambda src, metadata: make_ttgir(src, metadata, options, capability)
    elif language == Language.GLUON:
        stages["ttgir"] = lambda src, metadata: ttgir_opt(src, metadata, options, capability)
    stages["llir"] = lambda src, metadata: make_llir(src, metadata, options, capability, backend.target.arch)
    stages["ptx"] = lambda src, metadata: backend.make_ptx(src, metadata, options, backend.target.arch)
    stages["cubin"] = lambda src, metadata: backend.make_cubin(src, metadata, options, backend.target.arch)
