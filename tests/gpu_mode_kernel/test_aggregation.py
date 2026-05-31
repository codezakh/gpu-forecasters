"""Tests for ``aggregate_outcomes`` and ``aggregate_speedups``.

Pure functions — no Modal, no GPU. The aggregation logic is the
load-bearing pure computation between the Modal layer (which produces
``list[Option[KernelExecResult, ScoringError]]``) and the v2 provider
(which produces ``Evaluation[Observation]``). Each branch (compile-
fail, runtime-error, incorrect, infrastructure-failure, success) needs
a unit test because the search's reward and the mutation prompt's
feedback both depend on which arm gets selected.
"""

from __future__ import annotations

import math

import pytest
from pydantic import ConfigDict
from typing_extensions import TypedDict

from gpu_forecasters.gpu_mode_kernel.aggregation import (
    aggregate_outcomes,
    aggregate_speedups,
)
from gpu_forecasters.gpu_mode_kernel.core import (
    CaseSpeedupBase,
    CompileFailedFeedback,
    IncorrectFeedback,
    InfrastructureFailureFeedback,
    KernelExecResult,
    RuntimeErrorFeedback,
    SuccessFeedback,
)
from gpu_forecasters.gpu_mode_kernel.kernel_pack import KernelPack
from gpu_forecasters.kernelbench.isolated_scoring import ScoringError
from gpu_forecasters.typing_utils import Err, Ok, Option


_Outcome = Option[KernelExecResult, ScoringError]


# ---------------------------------------------------------------------------
# Fixtures: a tiny test pack with one shape parameter.
# ---------------------------------------------------------------------------


class FixtureTestArgs(TypedDict):
    shape: int


class FixtureCaseSpeedup(CaseSpeedupBase):
    model_config = ConfigDict(frozen=True)

    shape: int

    @classmethod
    def from_exec_result(
        cls,
        test_args,
        exec_result: KernelExecResult,
    ) -> "FixtureCaseSpeedup":
        speedup = (
            exec_result.ref_runtime_ns / exec_result.runtime_ns
            if exec_result.runtime_ns > 0
            else 0.0
        )
        return cls(
            shape=test_args["shape"],
            speedup=speedup,
            runtime_ns=exec_result.runtime_ns,
            ref_runtime_ns=exec_result.ref_runtime_ns,
        )

    def format_for_prompt(self) -> str:
        return f"shape={self.shape}: {self.speedup:.3f}x"


def _fixture_pack() -> KernelPack[FixtureTestArgs, FixtureCaseSpeedup]:
    return KernelPack(
        name="fixture",
        modal_app_name="fixture-app",
        correctness_cases=[],
        benchmark_cases=[],
        ref_kernel=lambda data: data,
        generate_input=lambda **kwargs: kwargs,
        check_implementation=lambda data, output: (True, ""),
        seed_kernel_code="",
        determinism_ctx=None,
        case_speedup_type=FixtureCaseSpeedup,
        kernel_description_body="",
    )


# ---------------------------------------------------------------------------
# aggregate_speedups
# ---------------------------------------------------------------------------


def test_aggregate_speedups_geomean() -> None:
    assert aggregate_speedups([2.0, 8.0], "geomean") == pytest.approx(4.0)


def test_aggregate_speedups_min() -> None:
    assert aggregate_speedups([2.0, 5.0, 1.5], "min") == 1.5


def test_aggregate_speedups_arith_mean() -> None:
    assert aggregate_speedups([1.0, 2.0, 3.0], "arith_mean") == pytest.approx(2.0)


def test_aggregate_speedups_geomean_three_factors() -> None:
    speedups = [1.5, 2.0, 3.0]
    expected = math.exp(sum(math.log(s) for s in speedups) / len(speedups))
    assert aggregate_speedups(speedups, "geomean") == pytest.approx(expected)


# ---------------------------------------------------------------------------
# aggregate_outcomes — short-circuit arms
# ---------------------------------------------------------------------------


def test_infrastructure_failure_short_circuits_first() -> None:
    pack = _fixture_pack()
    cases: list[FixtureTestArgs] = [{"shape": 1}, {"shape": 2}]
    outcomes: list[_Outcome] = [
        Err(ScoringError(reason="modal blew up", cause="...")),
        Ok(KernelExecResult(correct=True, runtime_ns=1.0, ref_runtime_ns=2.0)),
    ]
    result = aggregate_outcomes(
        outcomes=outcomes, test_cases=cases, pack=pack, aggregator="geomean"
    )
    assert isinstance(result.feedback, InfrastructureFailureFeedback)
    assert "modal blew up" in result.feedback.reason
    assert result.reward is None
    # The second outcome is never inspected.
    assert result.per_case_results == []
    assert result.n_correct == 0


def test_compile_failed_arm_short_circuits() -> None:
    pack = _fixture_pack()
    cases: list[FixtureTestArgs] = [{"shape": 1}, {"shape": 2}]
    outcomes: list[_Outcome] = [
        Ok(
            KernelExecResult(
                correct=False,
                runtime_ns=0.0,
                ref_runtime_ns=0.0,
                failure_kind="compile_failed",
                compilation_error="SyntaxError",
            )
        ),
        Ok(KernelExecResult(correct=True, runtime_ns=1.0, ref_runtime_ns=2.0)),
    ]
    result = aggregate_outcomes(
        outcomes=outcomes, test_cases=cases, pack=pack, aggregator="geomean"
    )
    assert isinstance(result.feedback, CompileFailedFeedback)
    assert result.reward is None
    # All outcomes were materialized into per_case_results before the
    # failure-arm dispatch — the failure arm walks results, not outcomes.
    assert len(result.per_case_results) == 2


def test_runtime_error_arm() -> None:
    pack = _fixture_pack()
    cases: list[FixtureTestArgs] = [{"shape": 1}]
    outcomes: list[_Outcome] = [
        Ok(
            KernelExecResult(
                correct=False,
                runtime_ns=0.0,
                ref_runtime_ns=0.0,
                failure_kind="runtime_error",
                runtime_error_name="RuntimeError",
                runtime_error="boom",
                traceback="tb",
            )
        ),
    ]
    result = aggregate_outcomes(
        outcomes=outcomes, test_cases=cases, pack=pack, aggregator="geomean"
    )
    assert isinstance(result.feedback, RuntimeErrorFeedback)
    assert result.feedback.runtime_error == "boom"
    assert result.reward is None


def test_incorrect_arm() -> None:
    pack = _fixture_pack()
    cases: list[FixtureTestArgs] = [{"shape": 1}]
    outcomes: list[_Outcome] = [
        Ok(
            KernelExecResult(
                correct=False,
                runtime_ns=0.0,
                ref_runtime_ns=0.0,
                failure_kind="incorrect",
                error_message="diff too big",
            )
        ),
    ]
    result = aggregate_outcomes(
        outcomes=outcomes, test_cases=cases, pack=pack, aggregator="geomean"
    )
    assert isinstance(result.feedback, IncorrectFeedback)
    assert result.feedback.error_message == "diff too big"


def test_success_arm_packs_speedup_and_reward() -> None:
    pack = _fixture_pack()
    cases: list[FixtureTestArgs] = [{"shape": 100}, {"shape": 200}]
    outcomes: list[_Outcome] = [
        Ok(KernelExecResult(correct=True, runtime_ns=1_000.0, ref_runtime_ns=2_000.0)),
        Ok(KernelExecResult(correct=True, runtime_ns=2_000.0, ref_runtime_ns=8_000.0)),
    ]
    result = aggregate_outcomes(
        outcomes=outcomes, test_cases=cases, pack=pack, aggregator="geomean"
    )
    assert isinstance(result.feedback, SuccessFeedback)
    assert result.feedback.aggregation_method == "geomean"
    # 2x and 4x → geomean = sqrt(8) ≈ 2.828
    assert result.feedback.aggregated_speedup == pytest.approx(math.sqrt(8.0))
    assert result.reward == pytest.approx(math.sqrt(8.0))
    assert result.n_correct == 2

    case_speedups = result.feedback.per_case_speedups
    assert len(case_speedups) == 2
    assert case_speedups[0].shape == 100
    assert case_speedups[0].speedup == pytest.approx(2.0)
    assert case_speedups[1].shape == 200
    assert case_speedups[1].speedup == pytest.approx(4.0)


def test_n_correct_counts_passing_only() -> None:
    pack = _fixture_pack()
    # Three outcomes — two correct, one runtime error. The failure
    # arm short-circuits, but n_correct still reflects the pass count
    # before the failing case appeared in the result list.
    cases: list[FixtureTestArgs] = [{"shape": 1}, {"shape": 2}, {"shape": 3}]
    outcomes: list[_Outcome] = [
        Ok(KernelExecResult(correct=True, runtime_ns=1.0, ref_runtime_ns=2.0)),
        Ok(KernelExecResult(correct=True, runtime_ns=1.0, ref_runtime_ns=2.0)),
        Ok(
            KernelExecResult(
                correct=False,
                runtime_ns=0.0,
                ref_runtime_ns=0.0,
                failure_kind="runtime_error",
                runtime_error_name="RuntimeError",
                runtime_error="boom",
                traceback="tb",
            )
        ),
    ]
    result = aggregate_outcomes(
        outcomes=outcomes, test_cases=cases, pack=pack, aggregator="geomean"
    )
    assert isinstance(result.feedback, RuntimeErrorFeedback)
    assert result.n_correct == 2
    assert len(result.per_case_results) == 3


