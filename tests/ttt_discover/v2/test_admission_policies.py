"""Unit tests for the admission policies.

The policies are pure functions of ``(candidate, parent)`` — no archive,
no I/O, no async. We assert on the returned typed ``AdmissionDecision``
for every outcome variant.
"""

from __future__ import annotations

from gpu_forecasters.trimul.core import (
    CaseSpeedup,
    CompileFailedFeedback,
    IncorrectFeedback,
    InfrastructureFailureFeedback,
    RuntimeErrorFeedback,
    SuccessFeedback,
)
from gpu_forecasters.ttt_discover.v2.admission_policies.insert_all import (
    InsertAllAdmissionPolicy,
)
from gpu_forecasters.ttt_discover.v2.admission_policies.success_only import (
    SuccessOnlyAdmissionPolicy,
)
from gpu_forecasters.ttt_discover.v2.domain.admission_decision import (
    AdmissionDecision,
    AdmitChild,
    CreditOnly,
)
from gpu_forecasters.ttt_discover.v2.domain.candidate import Candidate, CandidateId
from gpu_forecasters.ttt_discover.v2.domain.outcome import (
    ParseFailureFeedback,
    TriMulRLOutcome,
)


def _candidate(outcome: TriMulRLOutcome, reward: float = 0.0) -> Candidate:
    return Candidate(
        id=CandidateId("cid"),
        code="pass",
        timestep=0,
        parent_id=None,
        outcome=outcome,
        reward=reward,
    )


def _success() -> SuccessFeedback:
    return SuccessFeedback(
        aggregated_speedup=1.0,
        aggregation_method="geomean",
        per_case_speedups=[
            CaseSpeedup(
                seqlen=256,
                bs=2,
                dim=128,
                hiddendim=128,
                nomask=True,
                distribution="normal",
                speedup=1.0,
                runtime_ns=2_500_000.0,
                ref_runtime_ns=2_500_000.0,
            )
        ],
    )


_FAILURE_VARIANTS: list[TriMulRLOutcome] = [
    CompileFailedFeedback(compilation_error="oops"),
    RuntimeErrorFeedback(
        runtime_error="boom", runtime_error_name="ValueError", traceback="tb"
    ),
    IncorrectFeedback(error_message="max abs err 1"),
    ParseFailureFeedback(reason="no block"),
]


def test_success_only_admits_success() -> None:
    policy = SuccessOnlyAdmissionPolicy()
    decision: AdmissionDecision = policy.decide(_candidate(_success(), reward=1.0), None)
    assert isinstance(decision, AdmitChild)


def test_success_only_credits_only_on_every_failure_variant() -> None:
    policy = SuccessOnlyAdmissionPolicy()
    for outcome in _FAILURE_VARIANTS + [
        InfrastructureFailureFeedback(reason="modal died"),
    ]:
        decision = policy.decide(_candidate(outcome), None)
        assert isinstance(decision, CreditOnly), (
            f"expected CreditOnly for {outcome.kind}, got {type(decision).__name__}"
        )


def test_insert_all_admits_every_model_level_failure() -> None:
    policy = InsertAllAdmissionPolicy()
    for outcome in _FAILURE_VARIANTS:
        decision = policy.decide(_candidate(outcome), None)
        assert isinstance(decision, AdmitChild), (
            f"expected AdmitChild for {outcome.kind}, got {type(decision).__name__}"
        )


def test_insert_all_admits_success() -> None:
    policy = InsertAllAdmissionPolicy()
    decision = policy.decide(_candidate(_success(), reward=1.0), None)
    assert isinstance(decision, AdmitChild)


def test_insert_all_elides_infrastructure_failure() -> None:
    """Modal / harness crashes aren't the model's fault and carry no
    actionable signal — they get credit-only treatment even under the
    insert-all policy."""
    policy = InsertAllAdmissionPolicy()
    decision = policy.decide(
        _candidate(InfrastructureFailureFeedback(reason="modal died")), None
    )
    assert isinstance(decision, CreditOnly)
