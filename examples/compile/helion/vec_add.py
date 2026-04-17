# Tested with helion==1.0.0 torch==2.9.0+ triton==3.5.0+

import helion
import helion.language as hl
import torch

import triton_runner
triton_runner.configure_jit_backend()


@helion.kernel(config=helion.Config(block_sizes = [128, 128]))  # The @helion.kernel decorator marks this function for compilation
def example_add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # Host code: Standard PyTorch operations
    m, n = x.size()
    out = torch.empty_like(x)  # Allocate output tensor

    # The hl.tile loop defines the parallel execution structure
    for tile_m, tile_n in hl.tile([m, n]):
        # Device code: Everything inside the hl.tile loop runs on GPU
        out[tile_m, tile_n] = x[tile_m, tile_n] + y[tile_m, tile_n] # Simple element-wise addition expressed w/ pytorch ops

    return out  # Return the result back to the host

# Create some sample data
size = 1223, 2112
x = torch.randn(size, device="cuda")
y = torch.randn(size, device="cuda")

# Run the kernel
result = example_add(x, y)

# Verify result
expected = x + y
torch.testing.assert_close(result, expected)
print("✅ Results Match ✅")
