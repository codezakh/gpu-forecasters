"""Pure outcome → feedback aggregation.

Generalizes the per-kernel ``_build_evaluation`` body in the existing
v2 scoring providers (``trimul_modal._build_evaluation``,
``causal_conv1d_modal._build_evaluation``) — same control flow, no
kernel-specific logic. The provider layer wraps this into an
``Observation`` and adds wall-clock / sink bookkeeping.
"""

from __future__ import annotations

import math
from typing import Any, Generic, Literal, Mapping, assert_never, cast

from pydantic import BaseModel, ConfigDict

from gpu_forecasters.gpu_mode_kernel.core import (
    CaseSpeedupT,
    InfrastructureFailureFeedback,
    KernelExecResult,
    ObservationFeedback,
    SuccessFeedback,
    failure_feedback_from_exec_result,
)
from gpu_forecasters.gpu_mode_kernel.kernel_pack import KernelPack, TestArgsT
from gpu_forecasters.kernelbench.isolated_scoring import ScoringError
from gpu_forecasters.typing_utils import Option, is_ok


AggregationMethod = Literal["geomean", "min", "arith_mean"]


def aggregate_speedups(speedups: list[float], method: AggregationMethod) -> float:
    """Aggregate per-case speedup ratios into one scalar reward."""
    match method:
        case "geomean":
            return math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        case "min":
            return min(speedups)
        case "arith_mean":
            return sum(speedups) / len(speedups)
        case _:
            assert_never(method)


class AggregationResult(BaseModel, Generic[CaseSpeedupT]):
    """Return value of ``aggregate_outcomes``.

    Carries the union-typed feedback (in-band or infrastructure
    failure), the materialized exec results in the order seen, and a
    bookkeeping count of how many cases passed correctness. The
    provider uses ``n_correct`` for telemetry; the search itself only
    reads ``feedback`` and ``reward``.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    feedback: ObservationFeedback[CaseSpeedupT]
    per_case_results: list[KernelExecResult]
    n_correct: int
    reward: float | None


def aggregate_outcomes(
    outcomes: list[Option[KernelExecResult, ScoringError]],
    test_cases: list[TestArgsT],
    pack: KernelPack[TestArgsT, CaseSpeedupT],
    aggregator: AggregationMethod,
) -> AggregationResult[CaseSpeedupT]:
    """Map per-case Modal outcomes to one observation-shaped feedback.

    Pure — no logging, no I/O. The provider layer handles those.
    Mirrors the body of the existing ``_build_evaluation`` functions
    in the per-kernel v2 providers.

    Semantics:
    - First infrastructure failure short-circuits to
      ``InfrastructureFailureFeedback`` and stops walking outcomes.
    - First in-band failure (compile/runtime/incorrect) short-circuits
      to that arm.
    - Only when every case passed correctness do we build
      ``SuccessFeedback`` with ``aggregated_speedup`` as the reward.
    """
    exec_results: list[KernelExecResult] = []
    n_correct = 0

    for outcome in outcomes:
        if not is_ok(outcome):
            scoring_error = outcome.unwrap_err()
            return AggregationResult[CaseSpeedupT](
                feedback=InfrastructureFailureFeedback(
                    reason=scoring_error.reason
                ),
                per_case_results=exec_results,
                n_correct=n_correct,
                reward=None,
            )

        exec_result = outcome.unwrap()
        exec_results.append(exec_result)
        if exec_result.correct:
            n_correct += 1

    for exec_result in exec_results:
        if not exec_result.correct:
            failure = failure_feedback_from_exec_result(exec_result)
            return AggregationResult[CaseSpeedupT](
                feedback=failure,
                per_case_results=exec_results,
                n_correct=n_correct,
                reward=None,
            )

    per_case_speedups: list[CaseSpeedupT] = [
        # TypedDict is a runtime dict but isn't strictly a ``Mapping[str,
        # Any]`` to the type checker; the cast is a wire-shape coercion,
        # not a behavior change.
        pack.case_speedup_type.from_exec_result(
            cast(Mapping[str, Any], test_case), exec_result
        )
        for exec_result, test_case in zip(exec_results, test_cases)
    ]
    aggregated = aggregate_speedups(
        [c.speedup for c in per_case_speedups], aggregator
    )
    success = SuccessFeedback[CaseSpeedupT](
        aggregated_speedup=aggregated,
        aggregation_method=aggregator,
        per_case_speedups=per_case_speedups,
    )
    return AggregationResult[CaseSpeedupT](
        feedback=success,
        per_case_results=exec_results,
        n_correct=n_correct,
        reward=aggregated,
    )
