"""
Test to reproduce CUDA context poisoning issue.

This test demonstrates the bug described in:
docs/specs/gh004-fix-cuda-crash-poisoning-kernelbench-evals.md

When a CUDA kernel triggers an illegal memory access (cudaErrorIllegalAddress),
the CUDA context becomes "poisoned" and all subsequent CUDA operations in the
same process fail, even for unrelated valid kernels.

Test structure:
1. Score a known-good kernel (sanity check) - should PASS
2. Score a broken kernel that causes illegal memory access - should raise CUDA error
3. Try to score the known-good kernel again - currently FAILS (demonstrates poisoning)

IMPORTANT: This test intentionally poisons the CUDA context and will fail until
subprocess isolation is implemented. After the fix, step 3 should pass because
the broken kernel will be isolated in a subprocess.

The test is marked with @pytest.mark.poisoning so it can be run in isolation
and excluded from normal test runs with: pytest -m "not poisoning"
"""

import pytest
import torch
from pathlib import Path

from arid_badger.kernelbench.scoring import score_kernel


# Good reference kernel: simple PyTorch matmul (uses "Model" class)
GOOD_REFERENCE_KERNEL_CODE = """
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)


N = 128


def get_inputs():
    a = torch.rand(N, N)
    b = torch.rand(N, N)
    return [a, b]


def get_init_inputs():
    return []
"""


# Good mutated kernel: simple PyTorch matmul (uses "ModelNew" class)
GOOD_MUTATED_KERNEL_CODE = """
import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)
"""


# Broken kernel: Deliberately accesses nullptr to trigger CUDA illegal memory access
# Based on working element-wise add example, modified to cause runtime error
BROKEN_KERNEL_CODE = """
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline


# CUDA kernel that deliberately accesses null pointer
cuda_source = '''
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void broken_kernel(const float* a, const float* b, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Deliberately dereference nullptr to cause illegal memory access
        float* null_ptr = nullptr;
        out[idx] = null_ptr[idx];  // This will cause cudaErrorIllegalAddress
    }
}

torch::Tensor broken_add_cuda(torch::Tensor a, torch::Tensor b) {
    auto size = a.numel();
    auto out = torch::zeros_like(a);

    const int block_size = 256;
    const int num_blocks = (size + block_size - 1) / block_size;

    broken_kernel<<<num_blocks, block_size>>>(
        a.data_ptr<float>(),
        b.data_ptr<float>(),
        out.data_ptr<float>(),
        size
    );

    // Force synchronization to catch the error
    cudaDeviceSynchronize();

    return out;
}
'''

cpp_source = "torch::Tensor broken_add_cuda(torch::Tensor a, torch::Tensor b);"

# Load the inline CUDA extension
broken_module = load_inline(
    name='broken_cuda_module',
    cpp_sources=cpp_source,
    cuda_sources=cuda_source,
    functions=['broken_add_cuda'],
    verbose=False,
    extra_cflags=[''],
    extra_ldflags=['']
)


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # Use the broken CUDA kernel - this will cause illegal memory access
        result = broken_module.broken_add_cuda(a, b)
        return result
"""


@pytest.mark.poisoning
@pytest.mark.integration
@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_context_poisoning_from_illegal_memory_access() -> None:
    """
    Test that demonstrates CUDA context poisoning after illegal memory access.

    This test is EXPECTED TO FAIL until subprocess isolation is implemented.

    The test shows that:
    1. A valid kernel can be scored successfully (step 1)
    2. A broken kernel triggers illegal memory access (step 2)
    3. After poisoning, even the valid kernel from step 1 fails (step 3)

    After subprocess isolation is implemented, this test should be updated to
    verify that step 3 passes (the good kernel still works because the broken
    kernel was isolated in a subprocess).

    Related issue: docs/specs/gh004-fix-cuda-crash-poisoning-kernelbench-evals.md
    """

    # Step 1: Score good kernel (sanity check)
    # This should succeed without any issues
    print("\n=== Step 1: Scoring good kernel (sanity check) ===")
    result_1 = score_kernel(
        mutated_kernel_code=GOOD_MUTATED_KERNEL_CODE,
        reference_kernel_code=GOOD_REFERENCE_KERNEL_CODE,
        backend="cuda",
        precision="fp32",
        num_correct_trials=1,
        num_perf_trials=5,
        build_dir=Path("/tmp/test_cuda_poisoning_good_1"),
    )

    assert result_1.exec_result is not None, "First good kernel should compile"
    assert (
        result_1.exec_result.compiled
    ), f"First good kernel failed to compile. Metadata: {result_1.exec_result.metadata!r}"
    print(
        f"✓ Step 1 passed: Good kernel scored successfully (speedup={result_1.speedup})"
    )

    # Step 2: Score broken kernel (poison the context)
    # This should raise a CUDA error due to illegal memory access
    print("\n=== Step 2: Scoring broken kernel (expecting CUDA error) ===")
    with pytest.raises(
        (torch.cuda.OutOfMemoryError, RuntimeError, torch.AcceleratorError),
        match="(?i)(cuda|illegal|memory|accelerator)",
    ):
        score_kernel(
            mutated_kernel_code=BROKEN_KERNEL_CODE,
            reference_kernel_code=GOOD_REFERENCE_KERNEL_CODE,
            backend="cuda",
            precision="fp32",
            num_correct_trials=1,
            num_perf_trials=5,
            build_dir=Path("/tmp/test_cuda_poisoning_broken"),
        )
    print("✓ Step 2 passed: Broken kernel raised expected CUDA error")

    # Step 3: Try to score good kernel again (demonstrate poisoning)
    # EXPECTED BEHAVIOR (current bug): This will fail with CUDA error even though
    # it's the same valid kernel from step 1, because the CUDA context is poisoned.
    #
    # After subprocess isolation is implemented, this should pass.
    print("\n=== Step 3: Scoring good kernel again (demonstrating poisoning) ===")
    print(
        "NOTE: This step is EXPECTED TO FAIL until subprocess isolation is implemented."
    )
    print("      The CUDA context is poisoned from step 2, so even valid kernels fail.")

    result_3 = score_kernel(
        mutated_kernel_code=GOOD_MUTATED_KERNEL_CODE,
        reference_kernel_code=GOOD_REFERENCE_KERNEL_CODE,
        backend="cuda",
        precision="fp32",
        num_correct_trials=1,
        num_perf_trials=5,
        build_dir=Path("/tmp/test_cuda_poisoning_good_2"),
    )

    # If we reach here without error, the CUDA context was NOT poisoned
    # (which means subprocess isolation is working!)
    assert result_3.exec_result is not None, "Second good kernel should compile"
    assert (
        result_3.exec_result.compiled
    ), f"Second good kernel failed to compile. Metadata: {result_3.exec_result.metadata!r}"
    print(
        f"✓ Step 3 passed: Good kernel still works after broken kernel (speedup={result_3.speedup})"
    )
    print("✓ CUDA context isolation is working! The fix has been implemented.")
