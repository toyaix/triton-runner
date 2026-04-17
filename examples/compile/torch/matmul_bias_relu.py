import torch

import triton_runner
triton_runner.configure_jit_backend()



def matmul_bias_relu(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    return torch.relu(x @ weight + bias)


def main():
    try:
        compiled_matmul_bias_relu = torch.compile(matmul_bias_relu)
    except Exception as exc:
        print(f"torch.compile is unavailable in this environment: {exc}")
        return

    torch.manual_seed(0)
    device = "cuda"
    dtype = torch.bfloat16

    x = torch.randn((256, 512), device=device, dtype=dtype)
    weight = torch.randn((512, 384), device=device, dtype=dtype)
    bias = torch.randn((384,), device=device, dtype=dtype)

    expected = matmul_bias_relu(x, weight, bias)
    try:
        compiled = compiled_matmul_bias_relu(x, weight, bias)
    except Exception as exc:
        print(f"torch.compile failed to run this example: {exc}")
        return

    if torch.allclose(compiled.float(), expected.float(), atol=5e-2, rtol=5e-2):
        print("✅ torch.compile and eager match")
        print("If TorchInductor generated Triton kernels, Triton Runner logs should appear above.")
    else:
        max_diff = (compiled.float() - expected.float()).abs().max().item()
        print(f"❌ Results differ, max diff = {max_diff}")


if __name__ == "__main__":
    main()
