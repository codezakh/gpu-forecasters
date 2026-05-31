"""RolloutRecord Pydantic round-trip covering every outcome variant."""

from __future__ import annotations

import pytest

from gpu_forecasters.trimul.core import (
    CaseSpeedup,
    CompileFailedFeedback,
    IncorrectFeedback,
    InfrastructureFailureFeedback,
    RuntimeErrorFeedback,
    SuccessFeedback,
)
from gpu_forecasters.ttt_discover.v2.domain.candidate import CandidateId
from gpu_forecasters.ttt_discover.v2.domain.outcome import (
    ParseFailureFeedback,
    TriMulRLOutcome,
)
from gpu_forecasters.ttt_discover.v2.domain.records import RolloutRecord


def _make_record(outcome: TriMulRLOutcome) -> RolloutRecord:
    return RolloutRecord(
        step=3,
        group_index=1,
        rollout_index=2,
        timestamp_utc="2026-04-24T12:00:00+00:00",
        parent_id=CandidateId("parent-xyz"),
        candidate_id=CandidateId("child-abc"),
        task_prompt="TASK PROMPT",
        feedback_prompt="FEEDBACK PROMPT",
        raw_response="<|channel|>analysis<|message|>cot...<|channel|>final<|message|>final",
        parsed_code="def custom_kernel(data): ...",
        outcome=outcome,
        reward=0.5,
        prompt_tokens=10,
        response_tokens=100,
        sampling_time_s=1.25,
        eval_time_s=7.5,
    )


@pytest.mark.parametrize(
    "outcome",
    [
        ParseFailureFeedback(reason="no code block"),
        CompileFailedFeedback(compilation_error="SyntaxError: bad token"),
        RuntimeErrorFeedback(
            runtime_error_name="ValueError",
            runtime_error="expected 2D",
            traceback="Traceback (most recent call last):\n  ...",
        ),
        IncorrectFeedback(error_message="max abs err 1.2e-1"),
        SuccessFeedback(
            aggregated_speedup=1.8,
            aggregation_method="geomean",
            per_case_speedups=[
                CaseSpeedup(
                    seqlen=256,
                    bs=2,
                    dim=128,
                    hiddendim=128,
                    nomask=True,
                    distribution="normal",
                    speedup=1.5,
                    runtime_ns=3_000_000.0,
                    ref_runtime_ns=4_500_000.0,
                ),
            ],
        ),
        InfrastructureFailureFeedback(reason="modal crashed"),
    ],
)
def test_rollout_record_roundtrip(outcome: TriMulRLOutcome) -> None:
    record = _make_record(outcome)
    json_str = record.model_dump_json()
    parsed = RolloutRecord.model_validate_json(json_str)
    assert parsed == record
    assert parsed.outcome.kind == outcome.kind
