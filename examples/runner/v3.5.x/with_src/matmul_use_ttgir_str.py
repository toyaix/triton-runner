import triton
import triton.language as tl
import torch
import triton_runner

if triton.__version__ in ["3.2.0", "3.1.0", "3.0.0"]:
    DEVICE = torch.cuda.current_device()
else:
    DEVICE = triton.runtime.driver.active.get_active_torch_device()

matmul_ttgir_src = """
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [2, 16], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [4, 8], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:75", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @matmul_kernel(%a_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %b_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %c_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %M: i32 {tt.divisibility = 16 : i32}, %N: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}, %stride_am: i32 {tt.divisibility = 16 : i32}, %stride_bk: i32 {tt.divisibility = 16 : i32}, %stride_cm: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c16_i32 = arith.constant 16 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<16x16xf32, #blocked>
    %pid_n = tt.get_program_id x : i32
    %pid_m = tt.get_program_id y : i32
    %offs_m = arith.muli %pid_m, %c16_i32 : i32
    %offs_m_0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_m_1 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %offs_m_2 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %offs_m_3 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %offs_m_4 = tt.splat %offs_m : i32 -> tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_m_5 = tt.splat %offs_m : i32 -> tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %offs_m_6 = arith.addi %offs_m_4, %offs_m_0 : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %offs_m_7 = arith.addi %offs_m_5, %offs_m_1 : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked1}>>
    %offs_n = arith.muli %pid_n, %c16_i32 : i32
    %offs_n_8 = tt.splat %offs_n : i32 -> tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %offs_n_9 = tt.splat %offs_n : i32 -> tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %offs_n_10 = arith.addi %offs_n_8, %offs_m_2 : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>>
    %offs_n_11 = arith.addi %offs_n_9, %offs_m_3 : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %a_ptrs = tt.expand_dims %offs_m_6 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<16x1xi32, #blocked>
    %a_ptrs_12 = tt.expand_dims %offs_m_7 {axis = 1 : i32} : tensor<16xi32, #ttg.slice<{dim = 1, parent = #blocked1}>> -> tensor<16x1xi32, #blocked1>
    %a_ptrs_13 = tt.splat %stride_am : i32 -> tensor<16x1xi32, #blocked>
    %a_ptrs_14 = arith.muli %a_ptrs, %a_ptrs_13 : tensor<16x1xi32, #blocked>
    %a_ptrs_15 = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>, #blocked>
    %a_ptrs_16 = tt.addptr %a_ptrs_15, %a_ptrs_14 : tensor<16x1x!tt.ptr<f32>, #blocked>, tensor<16x1xi32, #blocked>
    %b_ptrs = tt.expand_dims %offs_n_10 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked1}>> -> tensor<1x16xi32, #blocked1>
    %b_ptrs_17 = tt.expand_dims %offs_n_11 {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xi32, #blocked>
    %b_ptrs_18 = tt.splat %b_ptr : !tt.ptr<f32> -> tensor<1x16x!tt.ptr<f32>, #blocked>
    %b_ptrs_19 = tt.addptr %b_ptrs_18, %b_ptrs_17 : tensor<1x16x!tt.ptr<f32>, #blocked>, tensor<1x16xi32, #blocked>
    %accumulator = scf.for %k = %c0_i32 to %K step %c1_i32 iter_args(%accumulator_32 = %cst) -> (tensor<16x16xf32, #blocked>)  : i32 {
      %a_ptrs_iter = tt.splat %k : i32 -> tensor<16x1xi32, #blocked>
      %a_ptrs_iter_33 = tt.addptr %a_ptrs_16, %a_ptrs_iter : tensor<16x1x!tt.ptr<f32>, #blocked>, tensor<16x1xi32, #blocked>
      %b_ptrs_iter = arith.muli %k, %stride_bk : i32
      %b_ptrs_iter_34 = tt.splat %b_ptrs_iter : i32 -> tensor<1x16xi32, #blocked>
      %b_ptrs_iter_35 = tt.addptr %b_ptrs_19, %b_ptrs_iter_34 : tensor<1x16x!tt.ptr<f32>, #blocked>, tensor<1x16xi32, #blocked>
      %a = tt.load %a_ptrs_iter_33 : tensor<16x1x!tt.ptr<f32>, #blocked>
      %b = tt.load %b_ptrs_iter_35 : tensor<1x16x!tt.ptr<f32>, #blocked>
      %accumulator_36 = tt.broadcast %a : tensor<16x1xf32, #blocked> -> tensor<16x16xf32, #blocked>
      %accumulator_37 = tt.broadcast %b : tensor<1x16xf32, #blocked> -> tensor<16x16xf32, #blocked>
      %accumulator_38 = arith.mulf %accumulator_36, %accumulator_37 : tensor<16x16xf32, #blocked>
      %accumulator_39 = arith.addf %accumulator_32, %accumulator_38 : tensor<16x16xf32, #blocked>
      scf.yield %accumulator_39 : tensor<16x16xf32, #blocked>
    }
    %c_ptrs = tt.splat %stride_cm : i32 -> tensor<16x1xi32, #blocked1>
    %c_ptrs_20 = arith.muli %a_ptrs_12, %c_ptrs : tensor<16x1xi32, #blocked1>
    %c_ptrs_21 = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>, #blocked1>
    %c_ptrs_22 = tt.addptr %c_ptrs_21, %c_ptrs_20 : tensor<16x1x!tt.ptr<f32>, #blocked1>, tensor<16x1xi32, #blocked1>
    %c_ptrs_23 = tt.broadcast %c_ptrs_22 : tensor<16x1x!tt.ptr<f32>, #blocked1> -> tensor<16x16x!tt.ptr<f32>, #blocked1>
    %c_ptrs_24 = tt.broadcast %b_ptrs : tensor<1x16xi32, #blocked1> -> tensor<16x16xi32, #blocked1>
    %c_ptrs_25 = tt.addptr %c_ptrs_23, %c_ptrs_24 : tensor<16x16x!tt.ptr<f32>, #blocked1>, tensor<16x16xi32, #blocked1>
    %c_mask = tt.splat %M : i32 -> tensor<16x1xi32, #blocked1>
    %c_mask_26 = arith.cmpi slt, %a_ptrs_12, %c_mask : tensor<16x1xi32, #blocked1>
    %c_mask_27 = tt.splat %N : i32 -> tensor<1x16xi32, #blocked1>
    %c_mask_28 = arith.cmpi slt, %b_ptrs, %c_mask_27 : tensor<1x16xi32, #blocked1>
    %c_mask_29 = tt.broadcast %c_mask_26 : tensor<16x1xi1, #blocked1> -> tensor<16x16xi1, #blocked1>
    %c_mask_30 = tt.broadcast %c_mask_28 : tensor<1x16xi1, #blocked1> -> tensor<16x16xi1, #blocked1>
    %c_mask_31 = arith.andi %c_mask_29, %c_mask_30 : tensor<16x16xi1, #blocked1>
    %0 = ttg.convert_layout %accumulator : tensor<16x16xf32, #blocked> -> tensor<16x16xf32, #blocked1>
    tt.store %c_ptrs_25, %0, %c_mask_31 : tensor<16x16x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}
"""

# @triton.jit
@triton_runner.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    pass


def matmul(a, b):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    M, K = a.shape
    K, N = b.shape
    # Allocates output.
    c = torch.empty((M, N), device=a.device, dtype=torch.float32)

    grid = lambda META: (triton.cdiv(N, META['BLOCK_SIZE_N']), triton.cdiv(M, META['BLOCK_SIZE_M']), )
    matmul_kernel[grid](
        a, b, c,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=16,
        BLOCK_SIZE_N=16,
        ttgir_src=matmul_ttgir_src,
    )
    return c


torch.manual_seed(0)
a = torch.randn((512, 1024), device=DEVICE, dtype=torch.float32)
b = torch.randn((1024, 256), device=DEVICE, dtype=torch.float32)
torch_output = torch.matmul(a, b)
triton_output = matmul(a, b)

if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=0):
    print("✅ Triton and Torch match")
else:
    print("❌ Triton and Torch differ")
