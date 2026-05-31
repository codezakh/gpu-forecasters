"""Modal-based EvaluationProvider for causal conv1d kernels.

Near-duplicate of ``trimul_modal.ModalProvider`` — same lifecycle,
same thread-pool ``batch_evaluate``, same invocation-sink shape. The
``CaseSpeedup`` constructor on the success path is the only place that
references kernel-specific shape fields (B, D, S, W); everything else
is generic and slated for the gh070-A task #3 extraction.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from types import TracebackType
from typing import Literal, Optional, assert_never, cast

from loguru import logger
from pydantic import BaseModel

from arid_badger.causal_conv1d.cases import CausalConv1dTestArgs
from arid_badger.causal_conv1d.core import (
    CaseSpeedup,
    CausalConv1dExecResult,
    CompileFailedFeedback,
    IncorrectFeedback,
    InfrastructureFailureFeedback,
    RuntimeErrorFeedback,
    SuccessFeedback,
    failure_feedback_from_exec_result,
)
from arid_badger.causal_conv1d.modal_scoring import (
    CausalConv1dScoringFn,
    modal_causal_conv1d_scoring_session,
)
from arid_badger.invocation_sink import InvocationSink, code_sha256
from arid_badger.typing_utils import implements, is_ok

from ..domain import Evaluation, EvaluationProvider
from .causal_conv1d import CausalConv1dObservation


# ---------------------------------------------------------------------------
# Aggregator (kernel-agnostic; will be lifted in #3)
# ---------------------------------------------------------------------------

AggregationMethod = Literal["geomean", "min", "arith_mean"]


def _aggregate_speedups(
    speedups: list[float],
    method: AggregationMethod,
) -> float:
    match method:
        case "geomean":
            return math.exp(sum(math.log(s) for s in speedups) / len(speedups))
        case "min":
            return min(speedups)
        case "arith_mean":
            return sum(speedups) / len(speedups)
        case _:
            assert_never(method)


# ---------------------------------------------------------------------------
# Invocation record
# ---------------------------------------------------------------------------


class ModalCausalConv1dEvaluationRecord(BaseModel, frozen=True):
    """Invocation record for a single causal conv1d Modal scoring call."""

    kind: Literal[
        "modal_causal_conv1d_evaluation"
    ] = "modal_causal_conv1d_evaluation"
    code_sha256: str
    wall_clock_seconds: float
    reward: float | None
    n_cases: int
    n_correct: int
    timestamp_utc: str


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class CausalConv1dModalProvider:
    """Evaluates causal conv1d candidates on Modal.

    Must be used as a context manager to manage the Modal session
    lifecycle. Each ``evaluate()`` call sends all test cases to a
    single Modal container and aggregates per-case speedups into one
    reward.
    """

    def __init__(
        self,
        test_cases: list[CausalConv1dTestArgs],
        aggregator: AggregationMethod = "geomean",
        gpu: str = "A100-80GB",
        max_repeats: int = 100,
        max_time_ns: float = 10e9,
        invocation_sink: Optional[InvocationSink] = None,
        max_batch_workers: int = 10,
    ) -> None:
        if not test_cases:
            raise ValueError("test_cases must be non-empty")
        self._test_cases = test_cases
        self._aggregator: AggregationMethod = aggregator
        self._gpu = gpu
        self._max_repeats = max_repeats
        self._max_time_ns = max_time_ns
        self._invocation_sink = invocation_sink
        self._max_batch_workers = max_batch_workers
        self._session_cm: Optional[AbstractContextManager[CausalConv1dScoringFn]] = None
        self._score_fn: Optional[CausalConv1dScoringFn] = None

    def __enter__(self) -> "CausalConv1dModalProvider":
        self._session_cm = modal_causal_conv1d_scoring_session(
            gpu=self._gpu,
            max_repeats=self._max_repeats,
            max_time_ns=self._max_time_ns,
        )
        self._score_fn = self._session_cm.__enter__()
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object,
    ) -> None:
        if self._session_cm is not None:
            self._session_cm.__exit__(
                cast(type[BaseException] | None, exc_type),
                cast(BaseException | None, exc_val),
                cast(TracebackType | None, exc_tb),
            )

    def evaluate(self, program_code: str) -> Evaluation[CausalConv1dObservation]:
        if self._score_fn is None:
            raise RuntimeError(
                "CausalConv1dModalProvider must be used as a context manager "
                "before calling evaluate()."
            )

        sha = code_sha256(program_code)
        logger.info(
            "Causal conv1d Modal eval launching: sha256={sha}", sha=sha[:8]
        )
        start_time_s = time.perf_counter()
        outcomes = self._score_fn(program_code, self._test_cases)
        wall_clock_seconds = time.perf_counter() - start_time_s

        exec_results: list[CausalConv1dExecResult] = []
        n_correct = 0

        for i, outcome in enumerate(outcomes):
            if not is_ok(outcome):
                scoring_error = outcome.unwrap_err()
                logger.warning(
                    "Causal conv1d Modal eval case {i} failed after "
                    "{elapsed:.1f}s: {reason}",
                    i=i,
                    elapsed=wall_clock_seconds,
                    reason=scoring_error.reason,
                )
                observation = CausalConv1dObservation(
                    feedback=InfrastructureFailureFeedback(
                        reason=scoring_error.reason
                    ),
                    per_case_results=exec_results,
                )
                self._record(sha, wall_clock_seconds, None, len(outcomes), n_correct)
                return Evaluation[CausalConv1dObservation](
                    observation=observation, reward=None
                )

            exec_result = outcome.unwrap()
            exec_results.append(exec_result)
            if exec_result.correct:
                n_correct += 1

        for exec_result in exec_results:
            if not exec_result.correct:
                feedback = failure_feedback_from_exec_result(exec_result)
                match feedback:
                    case CompileFailedFeedback():
                        outcome_name = "compile_failed"
                    case RuntimeErrorFeedback(runtime_error_name=name):
                        outcome_name = f"runtime_error({name})"
                    case IncorrectFeedback():
                        outcome_name = "incorrect"
                    case _:
                        assert_never(feedback)

                logger.info(
                    "Causal conv1d Modal eval done: reward=None, "
                    "elapsed={elapsed:.1f}s, sha256={sha}, outcome={outcome}, "
                    "n_correct={n_correct}/{n_cases}",
                    elapsed=wall_clock_seconds,
                    sha=sha[:8],
                    outcome=outcome_name,
                    n_correct=n_correct,
                    n_cases=len(exec_results),
                )
                observation = CausalConv1dObservation(
                    feedback=feedback,
                    per_case_results=exec_results,
                )
                self._record(
                    sha, wall_clock_seconds, None, len(exec_results), n_correct
                )
                return Evaluation[CausalConv1dObservation](
                    observation=observation, reward=None
                )

        per_case_speedups: list[CaseSpeedup] = []
        raw_speedups: list[float] = []
        for exec_result, test_case in zip(exec_results, self._test_cases):
            speedup = (
                exec_result.ref_runtime_ns / exec_result.runtime_ns
                if exec_result.runtime_ns > 0
                else 0.0
            )
            raw_speedups.append(speedup)
            per_case_speedups.append(
                CaseSpeedup(
                    B=test_case["B"],
                    D=test_case["D"],
                    S=test_case["S"],
                    W=test_case["W"],
                    speedup=speedup,
                    runtime_ns=exec_result.runtime_ns,
                    ref_runtime_ns=exec_result.ref_runtime_ns,
                )
            )

        aggregated_speedup = _aggregate_speedups(raw_speedups, self._aggregator)

        feedback = SuccessFeedback(
            aggregated_speedup=aggregated_speedup,
            aggregation_method=self._aggregator,
            per_case_speedups=per_case_speedups,
        )
        observation = CausalConv1dObservation(
            feedback=feedback,
            per_case_results=exec_results,
        )

        logger.info(
            "Causal conv1d Modal eval done: reward={reward}, "
            "elapsed={elapsed:.1f}s, sha256={sha}, outcome=success, "
            "n_cases={n_cases}",
            reward=f"{aggregated_speedup:.4f}",
            elapsed=wall_clock_seconds,
            sha=sha[:8],
            n_cases=len(exec_results),
        )

        self._record(
            sha,
            wall_clock_seconds,
            aggregated_speedup,
            len(exec_results),
            n_correct,
        )
        return Evaluation[CausalConv1dObservation](
            observation=observation, reward=aggregated_speedup
        )

    def batch_evaluate(
        self, program_codes: list[str]
    ) -> list[Evaluation[CausalConv1dObservation]]:
        if not program_codes:
            return []
        n_workers = min(len(program_codes), self._max_batch_workers)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            return list(executor.map(self.evaluate, program_codes))

    def _record(
        self,
        sha: str,
        wall_clock_seconds: float,
        reward: float | None,
        n_cases: int,
        n_correct: int,
    ) -> None:
        if self._invocation_sink is not None:
            self._invocation_sink.record(
                ModalCausalConv1dEvaluationRecord(
                    code_sha256=sha,
                    wall_clock_seconds=wall_clock_seconds,
                    reward=reward,
                    n_cases=n_cases,
                    n_correct=n_correct,
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                )
            )


implements(EvaluationProvider[CausalConv1dObservation])(CausalConv1dModalProvider)
