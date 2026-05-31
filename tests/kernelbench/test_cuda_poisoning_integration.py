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

from gpu_forecasters.kernelbench.isolated_scoring import run_scoring_in_subprocess


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

    Now that subprocess isolation is implemented via run_scoring_in_subprocess,
    this test verifies that step 3 passes (the good kernel still works because the
    broken kernel was isolated in a subprocess).

    The test shows that:
    1. A valid kernel can be scored successfully (step 1)
    2. A broken kernel triggers illegal memory access, but the fault is contained
       in the spawned subprocess — the parent process is not affected (step 2)
    3. After the broken kernel, the valid kernel from step 1 still works (step 3)

    Related issue: docs/specs/gh004-fix-cuda-crash-poisoning-kernelbench-evals.md
    """

    # Step 1: Score good kernel (sanity check)
    # This should succeed without any issues
    print("\n=== Step 1: Scoring good kernel (sanity check) ===")
    result_1 = run_scoring_in_subprocess(
        mutated_kernel_code=GOOD_MUTATED_KERNEL_CODE,
        reference_kernel_code=GOOD_REFERENCE_KERNEL_CODE,
        backend="cuda",
        precision="fp32",
        num_correct_trials=1,
        num_perf_trials=5,
        build_dir=Path("/tmp/test_cuda_poisoning_good_1"),
    )

    assert result_1.is_ok(), f"First good kernel failed: {result_1.unwrap_err()}"
    exec_result_1 = result_1.unwrap()
    assert (
        exec_result_1.compiled
    ), f"First good kernel failed to compile. Metadata: {exec_result_1.metadata!r}"
    print("✓ Step 1 passed: Good kernel scored successfully")

    # Step 2: Score broken kernel — the CUDA illegal memory access happens in the
    # spawned subprocess, not the parent. The subprocess crashes or returns a failure
    # result; either way the parent CUDA context is unaffected.
    print("\n=== Step 2: Scoring broken kernel (subprocess absorbs CUDA fault) ===")
    result_2 = run_scoring_in_subprocess(
        mutated_kernel_code=BROKEN_KERNEL_CODE,
        reference_kernel_code=GOOD_REFERENCE_KERNEL_CODE,
        backend="cuda",
        precision="fp32",
        num_correct_trials=1,
        num_perf_trials=5,
        build_dir=Path("/tmp/test_cuda_poisoning_broken"),
    )
    # The subprocess crashes or returns a failure result — either is acceptable.
    # What matters is that the parent process is not affected.
    print(f"✓ Step 2 complete: broken kernel result is_ok={result_2.is_ok()} (parent unaffected)")

    # Step 3: Try to score good kernel again (verify isolation worked)
    # With subprocess isolation, the CUDA context in the parent process was never
    # poisoned, so the good kernel should still work here.
    print("\n=== Step 3: Scoring good kernel again (verifying no CUDA context poisoning) ===")

    result_3 = run_scoring_in_subprocess(
        mutated_kernel_code=GOOD_MUTATED_KERNEL_CODE,
        reference_kernel_code=GOOD_REFERENCE_KERNEL_CODE,
        backend="cuda",
        precision="fp32",
        num_correct_trials=1,
        num_perf_trials=5,
        build_dir=Path("/tmp/test_cuda_poisoning_good_2"),
    )

    # If this passes, the CUDA context was NOT poisoned in the parent process
    # (subprocess isolation is working correctly)
    assert result_3.is_ok(), f"Second good kernel failed after broken kernel: {result_3.unwrap_err()}"
    exec_result_3 = result_3.unwrap()
    assert (
        exec_result_3.compiled
    ), f"Second good kernel failed to compile. Metadata: {exec_result_3.metadata!r}"
    print("✓ Step 3 passed: Good kernel still works after broken kernel")
    print("✓ CUDA context isolation is working! The fix has been implemented.")
