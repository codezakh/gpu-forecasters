"""Modal-based EvaluationProvider for TriMul kernels.

Parallel to ``kernelbench_modal.ModalProvider`` — same context-manager
lifecycle, same thread-pool ``batch_evaluate``, same invocation-sink
shape. Takes a ``test_cases: list[TriMulTestArgs]`` at construction;
each ``evaluate()`` call sends all cases to a single Modal container
and aggregates the per-case speedups into one reward.
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

from arid_badger.invocation_sink import InvocationSink, code_sha256
from arid_badger.trimul.cases import TriMulTestArgs
from arid_badger.trimul.core import (
    CaseSpeedup,
    CompileFailedFeedback,
    IncorrectFeedback,
    InfrastructureFailureFeedback,
    RuntimeErrorFeedback,
    SuccessFeedback,
    TriMulExecResult,
    failure_feedback_from_exec_result,
)
from arid_badger.trimul.modal_scoring import (
    TriMulScoringFn,
    modal_trimul_scoring_session,
)
from arid_badger.typing_utils import implements, is_ok

from ..domain import Evaluation, EvaluationProvider
from .trimul import TriMulObservation


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

AggregationMethod = Literal["geomean", "min", "arith_mean"]


def _aggregate_speedups(
    speedups: list[float],
    method: AggregationMethod,
) -> float:
    """Reduce per-case speedups to a single scalar reward."""
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


class ModalTriMulEvaluationRecord(BaseModel, frozen=True):
    """Invocation record for a single TriMul Modal scoring call."""

    kind: Literal["modal_trimul_evaluation"] = "modal_trimul_evaluation"
    code_sha256: str
    wall_clock_seconds: float
    reward: float | None
    n_cases: int
    n_correct: int
    timestamp_utc: str


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class TriMulModalProvider:
    """Evaluates TriMul candidates on Modal.

    Must be used as a context manager to manage the Modal session lifecycle.
    Each ``evaluate()`` call sends all test cases to a single Modal
    container and aggregates per-case speedups into one reward.
    """

    def __init__(
        self,
        test_cases: list[TriMulTestArgs],
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
        self._session_cm: Optional[AbstractContextManager[TriMulScoringFn]] = None
        self._score_fn: Optional[TriMulScoringFn] = None

    def __enter__(self) -> "TriMulModalProvider":
        self._session_cm = modal_trimul_scoring_session(
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

    def evaluate(self, program_code: str) -> Evaluation[TriMulObservation]:
        if self._score_fn is None:
            raise RuntimeError(
                "TriMulModalProvider must be used as a context manager before "
                "calling evaluate(). Use `with TriMulModalProvider(...) as p: ...`"
            )

        # Reward policy: any single incorrect or infrastructure-failed case
        # means reward=None for the whole candidate.  Only candidates that
        # pass every case get an aggregated speedup reward.
        sha = code_sha256(program_code)
        logger.info("TriMul Modal eval launching: sha256={sha}", sha=sha[:8])
        start_time_s = time.perf_counter()
        outcomes = self._score_fn(program_code, self._test_cases)
        wall_clock_seconds = time.perf_counter() - start_time_s

        # Unpack results, tracking per-case exec results and failures.
        exec_results: list[TriMulExecResult] = []
        n_correct = 0

        for i, outcome in enumerate(outcomes):
            # Infrastructure failure (Err slot) — bail immediately.
            if not is_ok(outcome):
                scoring_error = outcome.unwrap_err()
                logger.warning(
                    "TriMul Modal eval case {i} failed after {elapsed:.1f}s: {reason}",
                    i=i,
                    elapsed=wall_clock_seconds,
                    reason=scoring_error.reason,
                )
                observation = TriMulObservation(
                    feedback=InfrastructureFailureFeedback(
                        reason=scoring_error.reason
                    ),
                    per_case_results=exec_results,
                )
                self._record(sha, wall_clock_seconds, None, len(outcomes), n_correct)
                return Evaluation[TriMulObservation](
                    observation=observation, reward=None
                )

            exec_result = outcome.unwrap()
            exec_results.append(exec_result)
            if exec_result.correct:
                n_correct += 1

        # Check for any incorrect case — first failure determines feedback.
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
                    "TriMul Modal eval done: reward=None, elapsed={elapsed:.1f}s, "
                    "sha256={sha}, outcome={outcome}, n_correct={n_correct}/{n_cases}",
                    elapsed=wall_clock_seconds,
                    sha=sha[:8],
                    outcome=outcome_name,
                    n_correct=n_correct,
                    n_cases=len(exec_results),
                )
                observation = TriMulObservation(
                    feedback=feedback,
                    per_case_results=exec_results,
                )
                self._record(
                    sha, wall_clock_seconds, None, len(exec_results), n_correct
                )
                return Evaluation[TriMulObservation](
                    observation=observation, reward=None
                )

        # All correct — compute per-case speedups and aggregate.
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
                    seqlen=test_case["seqlen"],
                    bs=test_case["bs"],
                    dim=test_case["dim"],
                    hiddendim=test_case["hiddendim"],
                    nomask=test_case["nomask"],
                    distribution=test_case["distribution"],
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
        observation = TriMulObservation(
            feedback=feedback,
            per_case_results=exec_results,
        )

        logger.info(
            "TriMul Modal eval done: reward={reward}, elapsed={elapsed:.1f}s, "
            "sha256={sha}, outcome=success, n_cases={n_cases}",
            reward=f"{aggregated_speedup:.4f}",
            elapsed=wall_clock_seconds,
            sha=sha[:8],
            n_cases=len(exec_results),
        )

        self._record(
            sha, wall_clock_seconds, aggregated_speedup, len(exec_results), n_correct
        )
        return Evaluation[TriMulObservation](
            observation=observation, reward=aggregated_speedup
        )

    def batch_evaluate(
        self, program_codes: list[str]
    ) -> list[Evaluation[TriMulObservation]]:
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
                ModalTriMulEvaluationRecord(
                    code_sha256=sha,
                    wall_clock_seconds=wall_clock_seconds,
                    reward=reward,
                    n_cases=n_cases,
                    n_correct=n_correct,
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                )
            )


implements(EvaluationProvider[TriMulObservation])(TriMulModalProvider)
