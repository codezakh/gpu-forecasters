"""Unit tests for the goal-conditioned evaluation provider wrapper."""

from __future__ import annotations

import math
from concurrent.futures import Future
from typing import Self

import pytest

from gpu_forecasters.eval_dataset_builder.v1.domain import speedup_band_for_bin
from gpu_forecasters.eval_dataset_builder.v1.goal_conditioned_evaluation import (
    GoalConditionedEvaluationProvider,
    score_evaluation_against_target_bin,
)
from gpu_forecasters.gpu_mode_kernel.core import (
    CompileFailedFeedback,
    GpuModeKernelObservation,
    IncorrectFeedback,
    InfrastructureFailureFeedback,
    RuntimeErrorFeedback,
    SuccessFeedback,
)
from gpu_forecasters.gpu_mode_kernel.packs.trimul import TriMulCaseSpeedup
from gpu_forecasters.hill_climbing.domain import Evaluation
from gpu_forecasters.landscape_map.v1.domain import SpeedupBin


_BIN = SpeedupBin.HIGH_SPEEDUP
_MID = speedup_band_for_bin(_BIN).midpoint


def _success_evaluation(
    speedup: float,
) -> Evaluation[GpuModeKernelObservation[TriMulCaseSpeedup]]:
    feedback = SuccessFeedback[TriMulCaseSpeedup](
        aggregated_speedup=speedup,
        aggregation_method="geomean",
        per_case_speedups=[],
    )
    observation = GpuModeKernelObservation[TriMulCaseSpeedup](feedback=feedback)
    return Evaluation[GpuModeKernelObservation[TriMulCaseSpeedup]](
        observation=observation, reward=speedup
    )


def _failure_evaluations() -> list[Evaluation[GpuModeKernelObservation[TriMulCaseSpeedup]]]:
    feedbacks = [
        CompileFailedFeedback(compilation_error="boom"),
        RuntimeErrorFeedback(
            runtime_error_name="ValueError",
            runtime_error="bad",
            traceback="tb",
        ),
        IncorrectFeedback(error_message="mismatch"),
        InfrastructureFailureFeedback(reason="modal failed"),
    ]
    return [
        Evaluation[GpuModeKernelObservation[TriMulCaseSpeedup]](
            observation=GpuModeKernelObservation[TriMulCaseSpeedup](feedback=fb),
            reward=None,
        )
        for fb in feedbacks
    ]


# --- score function ---------------------------------------------------------


def test_score_at_midpoint_is_zero() -> None:
    score = score_evaluation_against_target_bin(
        _success_evaluation(_MID), target_bin=_BIN
    )
    assert score is not None
    assert math.isclose(score, 0.0, abs_tol=1e-9)


def test_score_above_band() -> None:
    score = score_evaluation_against_target_bin(
        _success_evaluation(2 * _MID), target_bin=_BIN
    )
    assert score is not None
    assert math.isclose(score, -math.log(2), abs_tol=1e-9)


def test_score_below_band() -> None:
    score = score_evaluation_against_target_bin(
        _success_evaluation(0.5 * _MID), target_bin=_BIN
    )
    assert score is not None
    assert math.isclose(score, -math.log(2), abs_tol=1e-9)


def test_score_failure_arms_are_none() -> None:
    for evaluation in _failure_evaluations():
        assert (
            score_evaluation_against_target_bin(evaluation, target_bin=_BIN) is None
        )


# --- wrapper class ---------------------------------------------------------


class _FakeInnerProvider:
    """Records lifecycle calls; ``submit`` returns a pre-resolved Future."""

    def __init__(
        self,
        evaluation: Evaluation[GpuModeKernelObservation[TriMulCaseSpeedup]] | None = None,
        exception: Exception | None = None,
    ) -> None:
        self.evaluation = evaluation
        self.exception = exception
        self.entered = False
        self.exited = False

    def __enter__(self) -> Self:
        self.entered = True
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.exited = True

    def submit(
        self, program_code: str
    ) -> Future[Evaluation[GpuModeKernelObservation[TriMulCaseSpeedup]]]:
        del program_code
        fut: Future[Evaluation[GpuModeKernelObservation[TriMulCaseSpeedup]]] = Future()
        if self.exception is not None:
            fut.set_exception(self.exception)
        else:
            assert self.evaluation is not None
            fut.set_result(self.evaluation)
        return fut


def test_wrapper_passes_through_observation_unchanged() -> None:
    inner_eval = _success_evaluation(_MID)
    inner = _FakeInnerProvider(evaluation=inner_eval)
    wrapper = GoalConditionedEvaluationProvider(
        inner_evaluation_provider=inner, target_bin=_BIN
    )
    out = wrapper.submit("code").result()
    assert out.observation is inner_eval.observation


def test_wrapper_substitutes_reward() -> None:
    inner_eval = _success_evaluation(2 * _MID)
    inner = _FakeInnerProvider(evaluation=inner_eval)
    wrapper = GoalConditionedEvaluationProvider(
        inner_evaluation_provider=inner, target_bin=_BIN
    )
    out = wrapper.submit("code").result()
    expected = score_evaluation_against_target_bin(inner_eval, target_bin=_BIN)
    assert out.reward is not None
    assert expected is not None
    assert math.isclose(out.reward, expected, abs_tol=1e-9)


def test_wrapper_propagates_exceptions() -> None:
    inner = _FakeInnerProvider(exception=RuntimeError("inner fault"))
    wrapper = GoalConditionedEvaluationProvider(
        inner_evaluation_provider=inner, target_bin=_BIN
    )
    fut = wrapper.submit("code")
    with pytest.raises(RuntimeError, match="inner fault"):
        _ = fut.result()


def test_lifecycle_is_no_op() -> None:
    """Wrapper does NOT manage the inner provider's lifecycle.
    BinFiller (the caller) owns it explicitly. The wrapper's
    ``__enter__``/``__exit__`` exist only to satisfy the
    ``AsyncEvaluationProvider`` Protocol shape."""
    inner = _FakeInnerProvider(evaluation=_success_evaluation(_MID))
    wrapper = GoalConditionedEvaluationProvider(
        inner_evaluation_provider=inner, target_bin=_BIN
    )
    with wrapper:
        assert not inner.entered
    assert not inner.exited


def test_failure_target_bin_rejected() -> None:
    inner = _FakeInnerProvider(evaluation=_success_evaluation(_MID))
    with pytest.raises(ValueError):
        _ = GoalConditionedEvaluationProvider(
            inner_evaluation_provider=inner,
            target_bin=SpeedupBin.FAILURE,
        )
