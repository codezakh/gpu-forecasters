"""Unit tests for the pure ``map_outcomes_to_rl_outcome`` function.

Avoids any real Modal calls — we build ``TriMulExecResult`` / ``Option``
lists directly and assert the mapped ``TriMulRLOutcome`` variant.
"""

from __future__ import annotations

from gpu_forecasters.kernelbench.isolated_scoring import ScoringError
from gpu_forecasters.trimul.cases import TriMulTestArgs
from gpu_forecasters.trimul.core import (
    CompileFailedFeedback,
    IncorrectFeedback,
    InfrastructureFailureFeedback,
    RuntimeErrorFeedback,
    SuccessFeedback,
    TriMulExecResult,
)
from gpu_forecasters.ttt_discover.v2.evaluator.modal_trimul import (
    map_outcomes_to_rl_outcome,
)
from gpu_forecasters.typing_utils import Err, Result


def _case(seqlen: int) -> TriMulTestArgs:
    return {
        "seqlen": seqlen,
        "bs": 1,
        "dim": 128,
        "hiddendim": 128,
        "seed": 1,
        "nomask": True,
        "distribution": "normal",
    }


def test_err_in_any_case_maps_to_infra_failure() -> None:
    cases = [_case(256), _case(512)]
    outcomes = [
        Result(
            TriMulExecResult(
                correct=True,
                runtime_ns=1_000_000.0,
                ref_runtime_ns=2_000_000.0,
                failure_kind="none",
            )
        ),
        Err(ScoringError(reason="modal died")),
    ]
    result = map_outcomes_to_rl_outcome(outcomes, cases)
    assert isinstance(result, InfrastructureFailureFeedback)
    assert "modal died" in result.reason


def test_compile_failed_maps_to_compile_feedback() -> None:
    cases = [_case(256)]
    outcomes = [
        Result(
            TriMulExecResult(
                correct=False,
                runtime_ns=0,
                ref_runtime_ns=0,
                failure_kind="compile_failed",
                compilation_error="SyntaxError",
            )
        )
    ]
    result = map_outcomes_to_rl_outcome(outcomes, cases)
    assert isinstance(result, CompileFailedFeedback)
    assert "SyntaxError" in result.compilation_error


def test_runtime_error_maps_to_runtime_feedback() -> None:
    cases = [_case(256)]
    outcomes = [
        Result(
            TriMulExecResult(
                correct=False,
                runtime_ns=0,
                ref_runtime_ns=0,
                failure_kind="runtime_error",
                runtime_error_name="ValueError",
                runtime_error="bad input",
                traceback="Traceback:\n  ...",
            )
        )
    ]
    result = map_outcomes_to_rl_outcome(outcomes, cases)
    assert isinstance(result, RuntimeErrorFeedback)
    assert result.runtime_error_name == "ValueError"


def test_incorrect_maps_to_incorrect_feedback() -> None:
    cases = [_case(256)]
    outcomes = [
        Result(
            TriMulExecResult(
                correct=False,
                runtime_ns=0,
                ref_runtime_ns=0,
                failure_kind="incorrect",
                error_message="max abs err",
            )
        )
    ]
    result = map_outcomes_to_rl_outcome(outcomes, cases)
    assert isinstance(result, IncorrectFeedback)


def test_all_correct_maps_to_success_with_per_case() -> None:
    cases = [_case(256), _case(512)]
    outcomes = [
        Result(
            TriMulExecResult(
                correct=True,
                runtime_ns=1_000_000.0,
                ref_runtime_ns=2_000_000.0,
                failure_kind="none",
            )
        ),
        Result(
            TriMulExecResult(
                correct=True,
                runtime_ns=4_000_000.0,
                ref_runtime_ns=2_000_000.0,
                failure_kind="none",
            )
        ),
    ]
    result = map_outcomes_to_rl_outcome(outcomes, cases)
    assert isinstance(result, SuccessFeedback)
    assert len(result.per_case_speedups) == 2
    assert result.aggregation_method == "geomean"


def test_empty_outcomes_maps_to_infra_failure() -> None:
    result = map_outcomes_to_rl_outcome([], [])
    assert isinstance(result, InfrastructureFailureFeedback)
