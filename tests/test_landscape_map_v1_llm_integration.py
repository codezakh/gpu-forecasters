"""
Integration test: full pipeline from KernelRuntimeQuery -> LLM call -> parsed KernelRuntimeEstimate.

Requires a live LLM API key. Run with: pytest -m integration
"""

import pytest

from arid_badger.landscape_map.v1.domain import (
    KernelImplementation,
    KernelRuntimeEstimate,
    KernelRuntimeQuery,
    KernelTaskInfo,
    LikertConfidence,
    LlmCallUsage,
    SpeedupBin,
)
from arid_badger.landscape_map.v1.llm_estimator import LlmSpeedupEstimator

# Minimal kernels that the LLM can reason about without needing real GPU compilation.
_PYTORCH_VECTOR_ADD = """\
import torch
import torch.nn as nn

class Model(nn.Module):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b
"""

_CUDA_VECTOR_ADD = """\
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void vector_add_kernel(const float* a, const float* b, float* out, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        out[idx] = a[idx] + b[idx];
    }
}

torch::Tensor vector_add(torch::Tensor a, torch::Tensor b) {
    auto out = torch::empty_like(a);
    int n = a.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vector_add_kernel<<<blocks, threads>>>(a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(), n);
    return out;
}
"""


@pytest.mark.integration
def test_llm_estimator_produces_valid_estimate() -> None:
    """Full pipeline: prompt rendering -> LLM call -> response parsing.

    Verifies that the prompt templates render correctly, the LLM returns a
    parseable response, and we can construct a valid KernelRuntimeEstimate.
    Does not assert a specific predicted_bin since that depends on the LLM.
    """
    query = KernelRuntimeQuery(
        task=KernelTaskInfo(op_name="vector_add", level_id=1, task_id=1),
        reference=KernelImplementation(
            kernel_name="pytorch_functional",
            code=_PYTORCH_VECTOR_ADD,
            runtime_ms=None,
        ),
        candidate=KernelImplementation(
            kernel_name="cuda_vector_add",
            code=_CUDA_VECTOR_ADD,
            runtime_ms=None,
        ),
        hardware=None,
    )

    estimator = LlmSpeedupEstimator(model_slug="gemini/gemini-2.0-flash", temperature=0.0)
    estimate, usage = estimator.estimate(query)

    # Result must be the right types
    assert isinstance(estimate, KernelRuntimeEstimate)
    assert isinstance(estimate.predicted_bin, SpeedupBin)

    # bin_confidences must cover exactly bins 1-8 with valid confidence values
    assert len(estimate.bin_confidences) == 8
    assert SpeedupBin.FAILURE not in estimate.bin_confidences
    for bin_val, confidence in estimate.bin_confidences.items():
        assert isinstance(bin_val, SpeedupBin)
        assert isinstance(confidence, LikertConfidence)

    # LLM must have produced a non-empty reasoning string
    assert estimate.reasoning.strip() != ""

    # If usage was returned, token counts must be positive
    if usage is not None:
        assert isinstance(usage, LlmCallUsage)
        assert usage.input_tokens > 0
        assert usage.output_tokens > 0
