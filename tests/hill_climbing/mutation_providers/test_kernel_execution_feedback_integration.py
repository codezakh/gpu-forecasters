"""Integration test for KernelExecutionFeedbackMutationProvider.

Exercises the real LiteLLM / Gemini path end-to-end, including the
`n=num_mutations` multi-candidate fan-in that the library version is built
around. Requires a live LLM API key loaded via `uv run --env-file .env`.

Run with: pytest -m integration
"""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from gpu_forecasters.hill_climbing.domain import Evaluation
from gpu_forecasters.hill_climbing.mutation_providers.kernel_execution_feedback import (
    KernelExecutionFeedbackMutationProvider,
    KernelExecutionFeedbackMutationRecord,
)
from gpu_forecasters.hill_climbing.scoring_providers.kernelbench import (
    KernelBenchObservation,
)
from gpu_forecasters.kernelbench.core import (
    InfrastructureFailureFeedback,
    SuccessFeedback,
)


_REFERENCE_KERNEL = """\
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a + b


def get_inputs():
    return [torch.randn(1024, device="cuda"), torch.randn(1024, device="cuda")]


def get_init_inputs():
    return []
"""


MODEL_SLUG = "gemini/gemini-2.5-flash"


class _ListSink:
    """In-memory InvocationSink that keeps a list of records."""

    def __init__(self) -> None:
        self.records: list[BaseModel] = []

    def record(self, payload: BaseModel) -> None:
        self.records.append(payload)


def _success_eval() -> Evaluation[KernelBenchObservation]:
    return Evaluation[KernelBenchObservation](
        observation=KernelBenchObservation(
            feedback=SuccessFeedback(
                runtime_us=100.0,
                ref_runtime_us=200.0,
                speedup=2.0,
            ),
        ),
        reward=2.0,
    )


def _infra_failure_eval() -> Evaluation[KernelBenchObservation]:
    return Evaluation[KernelBenchObservation](
        observation=KernelBenchObservation(
            feedback=InfrastructureFailureFeedback(reason="modal eviction"),
        ),
        reward=None,
    )


@pytest.mark.integration
def test_generate_mutations_success_feedback_end_to_end() -> None:
    """Real LLM call with SuccessFeedback, n=3.

    Verifies the feedback-prompt branch produces multiple parseable kernels
    in a single underlying `completion(n=3)` call, and that the invocation
    record aggregates correctly.
    """
    sink = _ListSink()
    provider = KernelExecutionFeedbackMutationProvider(
        reference_kernel_code=_REFERENCE_KERNEL,
        model_slug=MODEL_SLUG,
        invocation_sink=sink,
    )

    codes = provider.generate_mutations(
        program_code=_REFERENCE_KERNEL,
        num_mutations=3,
        evaluation=_success_eval(),
    )

    # The LLM must return at least one parseable candidate; we don't hard
    # require all 3 since candidate parsing can occasionally drop one.
    assert len(codes) >= 1
    assert len(codes) <= 3
    for code in codes:
        assert "ModelNew" in code, f"Expected ModelNew in candidate:\n{code}"
        assert "nn.Module" in code

    assert len(sink.records) == 1
    record = sink.records[0]
    assert isinstance(record, KernelExecutionFeedbackMutationRecord)
    assert record.num_mutations_requested == 3
    assert record.num_mutations_produced == len(codes)
    assert record.total_input_tokens > 0
    assert record.total_output_tokens > 0
    assert record.wall_clock_seconds > 0
    # Either the primary call sufficed (1 request) or one top-up fired (2).
    assert record.num_llm_requests in (1, 2)


@pytest.mark.integration
def test_generate_mutations_infrastructure_failure_end_to_end() -> None:
    """Real LLM call with InfrastructureFailureFeedback.

    Verifies the fallback-to-base-prompt branch: when execution feedback is
    unavailable, the provider issues the zero-shot base prompt and still
    returns parseable kernels.
    """
    sink = _ListSink()
    provider = KernelExecutionFeedbackMutationProvider(
        reference_kernel_code=_REFERENCE_KERNEL,
        model_slug=MODEL_SLUG,
        invocation_sink=sink,
    )

    codes = provider.generate_mutations(
        program_code=_REFERENCE_KERNEL,
        num_mutations=2,
        evaluation=_infra_failure_eval(),
    )

    assert len(codes) >= 1
    assert len(codes) <= 2
    for code in codes:
        assert "ModelNew" in code

    assert len(sink.records) == 1
    record = sink.records[0]
    assert isinstance(record, KernelExecutionFeedbackMutationRecord)
    assert record.num_mutations_requested == 2
    assert record.total_input_tokens > 0
    assert record.total_output_tokens > 0
