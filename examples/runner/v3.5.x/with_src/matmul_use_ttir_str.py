import triton
import triton.language as tl
import torch
import triton_runner

if triton.__version__ in ["3.2.0", "3.1.0", "3.0.0"]:
    DEVICE = torch.cuda.current_device()
else:
    DEVICE = triton.runtime.driver.active.get_active_torch_device()

matmul_ttir_src = """
module {
  tt.func public @matmul_kernel(%a_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %b_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %c_ptr: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %M: i32 {tt.divisibility = 16 : i32}, %N: i32 {tt.divisibility = 16 : i32}, %K: i32 {tt.divisibility = 16 : i32}, %stride_am: i32 {tt.divisibility = 16 : i32}, %stride_bk: i32 {tt.divisibility = 16 : i32}, %stride_cm: i32 {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %accumulator = arith.constant dense<0.000000e+00> : tensor<16x16xf32>
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c16_i32 = arith.constant 16 : i32
    %pid_n = tt.get_program_id x : i32
    %pid_m = tt.get_program_id y : i32
    %offs_m = arith.muli %pid_m, %c16_i32 : i32
    %offs_m_0 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %offs_m_1 = tt.splat %offs_m : i32 -> tensor<16xi32>
    %offs_m_2 = arith.addi %offs_m_1, %offs_m_0 : tensor<16xi32>
    %offs_n = arith.muli %pid_n, %c16_i32 : i32
    %offs_n_3 = tt.splat %offs_n : i32 -> tensor<16xi32>
    %offs_n_4 = arith.addi %offs_n_3, %offs_m_0 : tensor<16xi32>
    %a_ptrs = tt.expand_dims %offs_m_2 {axis = 1 : i32} : tensor<16xi32> -> tensor<16x1xi32>
    %a_ptrs_5 = tt.splat %stride_am : i32 -> tensor<16x1xi32>
    %a_ptrs_6 = arith.muli %a_ptrs, %a_ptrs_5 : tensor<16x1xi32>
    %a_ptrs_7 = tt.splat %a_ptr : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
    %a_ptrs_8 = tt.addptr %a_ptrs_7, %a_ptrs_6 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>
    %b_ptrs = tt.expand_dims %offs_n_4 {axis = 0 : i32} : tensor<16xi32> -> tensor<1x16xi32>
    %b_ptrs_9 = tt.splat %b_ptr : !tt.ptr<f32> -> tensor<1x16x!tt.ptr<f32>>
    %b_ptrs_10 = tt.addptr %b_ptrs_9, %b_ptrs : tensor<1x16x!tt.ptr<f32>>, tensor<1x16xi32>
    %accumulator_11 = scf.for %k = %c0_i32 to %K step %c1_i32 iter_args(%accumulator_24 = %accumulator) -> (tensor<16x16xf32>)  : i32 {
      %a_ptrs_iter = tt.splat %k : i32 -> tensor<16x1xi32>
      %a_ptrs_iter_25 = tt.addptr %a_ptrs_8, %a_ptrs_iter : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>
      %b_ptrs_iter = arith.muli %k, %stride_bk : i32
      %b_ptrs_iter_26 = tt.splat %b_ptrs_iter : i32 -> tensor<1x16xi32>
      %b_ptrs_iter_27 = tt.addptr %b_ptrs_10, %b_ptrs_iter_26 : tensor<1x16x!tt.ptr<f32>>, tensor<1x16xi32>
      %a = tt.load %a_ptrs_iter_25 : tensor<16x1x!tt.ptr<f32>>
      %b = tt.load %b_ptrs_iter_27 : tensor<1x16x!tt.ptr<f32>>
      %accumulator_28 = tt.broadcast %a : tensor<16x1xf32> -> tensor<16x16xf32>
      %accumulator_29 = tt.broadcast %b : tensor<1x16xf32> -> tensor<16x16xf32>
      %accumulator_30 = arith.mulf %accumulator_28, %accumulator_29 : tensor<16x16xf32>
      %accumulator_31 = arith.addf %accumulator_24, %accumulator_30 : tensor<16x16xf32>
      scf.yield %accumulator_31 : tensor<16x16xf32>
    }
    %c_ptrs = tt.splat %stride_cm : i32 -> tensor<16x1xi32>
    %c_ptrs_12 = arith.muli %a_ptrs, %c_ptrs : tensor<16x1xi32>
    %c_ptrs_13 = tt.splat %c_ptr : !tt.ptr<f32> -> tensor<16x1x!tt.ptr<f32>>
    %c_ptrs_14 = tt.addptr %c_ptrs_13, %c_ptrs_12 : tensor<16x1x!tt.ptr<f32>>, tensor<16x1xi32>
    %c_ptrs_15 = tt.broadcast %c_ptrs_14 : tensor<16x1x!tt.ptr<f32>> -> tensor<16x16x!tt.ptr<f32>>
    %c_ptrs_16 = tt.broadcast %b_ptrs : tensor<1x16xi32> -> tensor<16x16xi32>
    %c_ptrs_17 = tt.addptr %c_ptrs_15, %c_ptrs_16 : tensor<16x16x!tt.ptr<f32>>, tensor<16x16xi32>
    %c_mask = tt.splat %M : i32 -> tensor<16x1xi32>
    %c_mask_18 = arith.cmpi slt, %a_ptrs, %c_mask : tensor<16x1xi32>
    %c_mask_19 = tt.splat %N : i32 -> tensor<1x16xi32>
    %c_mask_20 = arith.cmpi slt, %b_ptrs, %c_mask_19 : tensor<1x16xi32>
    %c_mask_21 = tt.broadcast %c_mask_18 : tensor<16x1xi1> -> tensor<16x16xi1>
    %c_mask_22 = tt.broadcast %c_mask_20 : tensor<1x16xi1> -> tensor<16x16xi1>
    %c_mask_23 = arith.andi %c_mask_21, %c_mask_22 : tensor<16x16xi1>
    tt.store %c_ptrs_17, %accumulator_11, %c_mask_23 : tensor<16x16x!tt.ptr<f32>>
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
        ttir_src=matmul_ttir_src,
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
