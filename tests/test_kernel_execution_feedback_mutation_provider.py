"""Tests for KernelExecutionFeedbackMutationProvider.

Only tests real domain logic we wrote: the feedback-branch selection, the
single-call-plus-topup fanout strategy, and the invocation-sink record shape.
Does not test Pydantic construction / litellm internals.
"""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock, patch

from arid_badger.hill_climbing.domain import Evaluation
from arid_badger.hill_climbing.mutation_providers.kernel_execution_feedback import (
    KernelExecutionFeedbackMutationProvider,
    KernelExecutionFeedbackMutationRecord,
)
from arid_badger.hill_climbing.scoring_providers.kernelbench import (
    KernelBenchObservation,
)
from arid_badger.invocation_sink import InvocationSink
from arid_badger.kernelbench.core import (
    InfrastructureFailureFeedback,
    SuccessFeedback,
)


REFERENCE_CODE = """
import torch
import torch.nn as nn
class Model(nn.Module):
    def forward(self, x):
        return x
"""

CHILD_1 = "```python\nclass ModelNew(nn.Module):\n    def forward(self, x):\n        return x + 1\n```"
CHILD_2 = "```python\nclass ModelNew(nn.Module):\n    def forward(self, x):\n        return x * 2\n```"
CHILD_3 = "```python\nclass ModelNew(nn.Module):\n    def forward(self, x):\n        return x - 1\n```"
UNPARSEABLE = "no code block here, sorry"


def _make_response(contents: list[str], input_tokens: int = 100, output_tokens: int = 50) -> MagicMock:
    response = MagicMock()
    response.choices = [MagicMock() for _ in contents]
    for choice, content in zip(response.choices, contents):
        choice.message.content = content
    response.usage.prompt_tokens = input_tokens
    response.usage.completion_tokens = output_tokens
    return response


def _success_eval() -> Evaluation[KernelBenchObservation]:
    feedback = SuccessFeedback(
        runtime_us=100.0,
        ref_runtime_us=200.0,
        speedup=2.0,
    )
    return Evaluation[KernelBenchObservation](
        observation=KernelBenchObservation(feedback=feedback),
        reward=2.0,
    )


def _infra_failure_eval() -> Evaluation[KernelBenchObservation]:
    return Evaluation[KernelBenchObservation](
        observation=KernelBenchObservation(
            feedback=InfrastructureFailureFeedback(reason="modal timeout"),
        ),
        reward=None,
    )


def _make_provider(sink: InvocationSink | None = None) -> KernelExecutionFeedbackMutationProvider:
    return KernelExecutionFeedbackMutationProvider(
        reference_kernel_code=REFERENCE_CODE,
        model_slug="gemini/gemini-2.5-flash",
        invocation_sink=sink,
    )


class TestGenerateMutations:
    def test_single_call_n_equals_num_mutations(self) -> None:
        provider = _make_provider()
        response = _make_response([CHILD_1, CHILD_2, CHILD_3])

        with patch(
            "arid_badger.hill_climbing.mutation_providers.kernel_execution_feedback.litellm.completion",
            return_value=response,
        ) as mock_completion:
            codes = provider.generate_mutations("prev code", 3, _success_eval())

        assert mock_completion.call_count == 1
        assert mock_completion.call_args.kwargs["n"] == 3
        assert len(codes) == 3

    def test_success_feedback_uses_feedback_prompt(self) -> None:
        provider = _make_provider()
        response = _make_response([CHILD_1])

        with patch(
            "arid_badger.hill_climbing.mutation_providers.kernel_execution_feedback.litellm.completion",
            return_value=response,
        ) as mock_completion:
            provider.generate_mutations("prev code", 1, _success_eval())

        prompt = mock_completion.call_args.kwargs["messages"][0]["content"]
        # Feedback prompt for SuccessFeedback embeds the speedup line.
        assert "Speedup: 2.0000x" in prompt
        assert "prev code" in prompt

    def test_infrastructure_failure_uses_base_prompt(self) -> None:
        provider = _make_provider()
        response = _make_response([CHILD_1])

        with patch(
            "arid_badger.hill_climbing.mutation_providers.kernel_execution_feedback.litellm.completion",
            return_value=response,
        ) as mock_completion:
            provider.generate_mutations("prev code", 1, _infra_failure_eval())

        prompt = mock_completion.call_args.kwargs["messages"][0]["content"]
        # Base prompt does NOT mention the previous kernel or an evaluation.
        assert "latest generation" not in prompt
        assert "prev code" not in prompt

    def test_topup_fills_shortfall(self) -> None:
        provider = _make_provider()
        # Primary call: 2 of 3 candidates parse.
        primary = _make_response([CHILD_1, UNPARSEABLE, CHILD_2])
        # Top-up call for deficit=1: returns 1 parseable.
        topup = _make_response([CHILD_3])

        with patch(
            "arid_badger.hill_climbing.mutation_providers.kernel_execution_feedback.litellm.completion",
            side_effect=[primary, topup],
        ) as mock_completion:
            codes = provider.generate_mutations("prev code", 3, _success_eval())

        assert mock_completion.call_count == 2
        assert mock_completion.call_args_list[0].kwargs["n"] == 3
        assert mock_completion.call_args_list[1].kwargs["n"] == 1
        assert len(codes) == 3

    def test_topup_is_single_attempt_only(self) -> None:
        provider = _make_provider()
        # Primary returns 1 parseable of 3.
        primary = _make_response([CHILD_1, UNPARSEABLE, UNPARSEABLE])
        # Top-up also under-delivers: still returns 0 parseable of 2 requested.
        topup = _make_response([UNPARSEABLE, UNPARSEABLE])

        with patch(
            "arid_badger.hill_climbing.mutation_providers.kernel_execution_feedback.litellm.completion",
            side_effect=[primary, topup],
        ) as mock_completion:
            codes = provider.generate_mutations("prev code", 3, _success_eval())

        # Only one top-up attempt — total 2 calls, partial batch accepted.
        assert mock_completion.call_count == 2
        assert len(codes) == 1

    def test_primary_request_exception_returns_empty(self) -> None:
        provider = _make_provider()

        with patch(
            "arid_badger.hill_climbing.mutation_providers.kernel_execution_feedback.litellm.completion",
            side_effect=[RuntimeError("api down"), _make_response([CHILD_1, CHILD_2, CHILD_3])],
        ) as mock_completion:
            codes = provider.generate_mutations("prev code", 3, _success_eval())

        # Primary raised → 0 codes → top-up asks for 3 → returns 3.
        assert mock_completion.call_count == 2
        assert len(codes) == 3


class TestInvocationSink:
    def test_records_aggregate_across_primary_and_topup(self) -> None:
        sink = MagicMock(spec=InvocationSink)
        provider = _make_provider(sink=sink)
        primary = _make_response([CHILD_1, UNPARSEABLE], input_tokens=100, output_tokens=40)
        topup = _make_response([CHILD_2], input_tokens=90, output_tokens=15)

        with patch(
            "arid_badger.hill_climbing.mutation_providers.kernel_execution_feedback.litellm.completion",
            side_effect=[primary, topup],
        ):
            provider.generate_mutations("prev code", 2, _success_eval())

        assert sink.record.call_count == 1
        record = cast(
            KernelExecutionFeedbackMutationRecord, sink.record.call_args.args[0]
        )
        assert record.num_mutations_requested == 2
        assert record.num_mutations_produced == 2
        assert record.num_llm_requests == 2
        assert record.total_input_tokens == 190
        assert record.total_output_tokens == 55
        assert len(record.child_code_sha256s) == 2

    def test_no_record_when_sink_none(self) -> None:
        provider = _make_provider(sink=None)
        response = _make_response([CHILD_1])

        with patch(
            "arid_badger.hill_climbing.mutation_providers.kernel_execution_feedback.litellm.completion",
            return_value=response,
        ):
            # Should not raise.
            provider.generate_mutations("prev code", 1, _success_eval())
