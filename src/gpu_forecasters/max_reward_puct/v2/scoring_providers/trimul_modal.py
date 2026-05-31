"""Native v2 TriMul Modal evaluation provider.

Per-candidate contract: one ``submit(code)`` = exactly one remote Modal
call producing exactly one ``Evaluation[TriMulObservation]``. Dispatch
is non-blocking: we call ``Function.spawn(...)`` (which returns
immediately with a ``FunctionCall`` handle) and hand the caller a
``concurrent.futures.Future`` backed by an internal thread that waits
on ``FunctionCall.get()``.

No batch-across-candidates abstraction. Concurrency at the
candidate level is whatever the driver asks for — the provider can
serve many ``submit`` calls in flight simultaneously (bounded by an
internal worker pool).

Reward aggregation and feedback shaping are ported from v1's
``TriMulModalProvider.evaluate``. The logic is pure over the Modal
outcome list; duplicating it keeps v2 independent of v1's batch-
oriented surface.
"""

from __future__ import annotations

import math
import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from types import TracebackType
from typing import Literal, Optional, Self, assert_never, cast

from loguru import logger
from pydantic import BaseModel

from gpu_forecasters.hill_climbing.domain import Evaluation
from gpu_forecasters.hill_climbing.scoring_providers.trimul import TriMulObservation
from gpu_forecasters.invocation_sink import InvocationSink, code_sha256
from gpu_forecasters.kernelbench.isolated_scoring import ScoringError
from gpu_forecasters.trimul.cases import TriMulTestArgs
from gpu_forecasters.trimul.core import (
    CaseSpeedup,
    CompileFailedFeedback,
    IncorrectFeedback,
    InfrastructureFailureFeedback,
    RuntimeErrorFeedback,
    SuccessFeedback,
    TriMulExecResult,
    failure_feedback_from_exec_result,
)
from gpu_forecasters.trimul.modal_scoring import (
    ModalTriMulBenchmarker,
    app as modal_app,
)
from gpu_forecasters.typing_utils import Option, is_ok


AggregationMethod = Literal["geomean", "min", "arith_mean"]


def _aggregate_speedups(
    speedups: list[float], method: AggregationMethod
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


class ModalTriMulEvaluationRecord(BaseModel, frozen=True):
    """Invocation record for a single TriMul Modal scoring call."""

    kind: Literal["modal_trimul_evaluation_v2"] = "modal_trimul_evaluation_v2"
    code_sha256: str
    wall_clock_seconds: float
    reward: float | None
    n_cases: int
    n_correct: int
    timestamp_utc: str


class TriMulModalProvider:
    """Async per-candidate TriMul evaluation provider over Modal.

    Implements ``AsyncEvaluationProvider[TriMulObservation]``.

    Lifecycle: ``__enter__`` opens a Modal ``app.run()`` session and
    starts an internal worker pool that drains ``FunctionCall.get()``
    futures. ``__exit__`` shuts down the pool and exits the session.
    """

    def __init__(
        self,
        test_cases: list[TriMulTestArgs],
        *,
        aggregator: AggregationMethod = "geomean",
        gpu: str = "A100-80GB",
        max_repeats: int = 100,
        max_time_ns: float = 10e9,
        max_in_flight: int = 10,
        max_containers: int = 10,
        get_timeout_s: float = 1200.0,
        invocation_sink: Optional[InvocationSink] = None,
    ) -> None:
        if not test_cases:
            raise ValueError("test_cases must be non-empty")
        if max_in_flight < 1:
            raise ValueError("max_in_flight must be >= 1")
        if max_containers < 1:
            raise ValueError("max_containers must be >= 1")
        self._test_cases = test_cases
        self._aggregator: AggregationMethod = aggregator
        self._gpu = gpu
        self._max_repeats = max_repeats
        self._max_time_ns = max_time_ns
        self._max_in_flight = max_in_flight
        self._max_containers = max_containers
        self._get_timeout_s = get_timeout_s
        self._invocation_sink = invocation_sink

        self._app_run_cm: AbstractContextManager[object] | None = None
        self._benchmarker_cls: type[ModalTriMulBenchmarker] | None = None
        self._waiter_pool: ThreadPoolExecutor | None = None

    # --- Lifecycle -----------------------------------------------------

    def __enter__(self) -> Self:
        self._app_run_cm = modal_app.run()
        _ = self._app_run_cm.__enter__()
        self._benchmarker_cls = ModalTriMulBenchmarker.with_options(  # ty: ignore[unresolved-attribute]  # pyright: ignore[reportAttributeAccessIssue]
            gpu=self._gpu, max_containers=self._max_containers
        )
        self._waiter_pool = ThreadPoolExecutor(
            max_workers=self._max_in_flight,
            thread_name_prefix="trimul-modal-waiter",
        )
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object,
    ) -> None:
        if self._waiter_pool is not None:
            self._waiter_pool.shutdown(wait=True)
            self._waiter_pool = None
        if self._app_run_cm is not None:
            self._app_run_cm.__exit__(
                cast(type[BaseException] | None, exc_type),
                cast(BaseException | None, exc_val),
                cast(TracebackType | None, exc_tb),
            )
            self._app_run_cm = None
        self._benchmarker_cls = None

    # --- Submit --------------------------------------------------------

    def submit(self, program_code: str) -> Future[Evaluation[TriMulObservation]]:
        if self._benchmarker_cls is None or self._waiter_pool is None:
            raise RuntimeError(
                "TriMulModalProvider must be entered as a context manager "
                "before submit()."
            )
        start_time_s = time.perf_counter()
        # Non-blocking dispatch: spawn immediately returns a
        # FunctionCall handle. The remote work begins on Modal right
        # away; the wait happens inside the waiter thread.
        function_call = self._benchmarker_cls().evaluate_candidate.spawn(
            mutated_kernel_code=program_code,
            test_cases=[dict(tc) for tc in self._test_cases],
            max_repeats=self._max_repeats,
            max_time_ns=self._max_time_ns,
        )
        return self._waiter_pool.submit(
            self._wait_and_wrap, program_code, function_call, start_time_s
        )

    def _wait_and_wrap(
        self,
        program_code: str,
        function_call,  # type: ignore[no-untyped-def]
        start_time_s: float,
    ) -> Evaluation[TriMulObservation]:
        sha = code_sha256(program_code)
        try:
            outcomes: list[Option[TriMulExecResult, ScoringError]] = (
                function_call.get(timeout=self._get_timeout_s)
            )
        except Exception as exc:
            wall = time.perf_counter() - start_time_s
            logger.warning(
                "TriMul Modal eval failed (sha={sha}, elapsed={e:.1f}s): {err}",
                sha=sha[:8],
                e=wall,
                err=exc,
            )
            self._record(sha, wall, None, 0, 0)
            observation = TriMulObservation(
                feedback=InfrastructureFailureFeedback(
                    reason=f"modal call failed: {type(exc).__name__}: {exc}"
                ),
                per_case_results=[],
            )
            return Evaluation[TriMulObservation](
                observation=observation, reward=None
            )

        wall = time.perf_counter() - start_time_s
        return _build_evaluation(
            sha=sha,
            outcomes=outcomes,
            test_cases=self._test_cases,
            aggregator=self._aggregator,
            wall_clock_seconds=wall,
            sink=self._invocation_sink,
            record=self._record,
        )

    # --- Sink bookkeeping ---------------------------------------------

    def _record(
        self,
        sha: str,
        wall: float,
        reward: float | None,
        n_cases: int,
        n_correct: int,
    ) -> None:
        if self._invocation_sink is None:
            return
        self._invocation_sink.record(
            ModalTriMulEvaluationRecord(
                code_sha256=sha,
                wall_clock_seconds=wall,
                reward=reward,
                n_cases=n_cases,
                n_correct=n_correct,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
            )
        )


# ---------------------------------------------------------------------------
# Pure outcome → evaluation shaping (module-level so it can be unit-tested).
# ---------------------------------------------------------------------------


def _build_evaluation(
    *,
    sha: str,
    outcomes: list[Option[TriMulExecResult, ScoringError]],
    test_cases: list[TriMulTestArgs],
    aggregator: AggregationMethod,
    wall_clock_seconds: float,
    sink: InvocationSink | None,
    record,  # type: ignore[no-untyped-def]
) -> Evaluation[TriMulObservation]:
    """Map Modal per-case outcomes to an ``Evaluation``. Pure
    (`sink`/`record` are passed in; this helper doesn't decide policy
    on sink presence — that's the caller's concern)."""
    del sink  # unused; the caller passes ``record`` already-bound.

    exec_results: list[TriMulExecResult] = []
    n_correct = 0

    for i, outcome in enumerate(outcomes):
        if not is_ok(outcome):
            scoring_error = outcome.unwrap_err()
            logger.warning(
                "TriMul Modal eval case {i} infrastructure failure: {reason}",
                i=i,
                reason=scoring_error.reason,
            )
            observation = TriMulObservation(
                feedback=InfrastructureFailureFeedback(
                    reason=scoring_error.reason
                ),
                per_case_results=exec_results,
            )
            record(sha, wall_clock_seconds, None, len(outcomes), n_correct)
            return Evaluation[TriMulObservation](
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
                "TriMul Modal eval: reward=None, elapsed={e:.1f}s, sha={sha}, "
                "outcome={o}, n_correct={nc}/{n}",
                e=wall_clock_seconds,
                sha=sha[:8],
                o=outcome_name,
                nc=n_correct,
                n=len(exec_results),
            )
            observation = TriMulObservation(
                feedback=feedback, per_case_results=exec_results
            )
            record(sha, wall_clock_seconds, None, len(exec_results), n_correct)
            return Evaluation[TriMulObservation](
                observation=observation, reward=None
            )

    per_case_speedups: list[CaseSpeedup] = []
    raw_speedups: list[float] = []
    for exec_result, test_case in zip(exec_results, test_cases):
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

    aggregated = _aggregate_speedups(raw_speedups, aggregator)
    feedback = SuccessFeedback(
        aggregated_speedup=aggregated,
        aggregation_method=aggregator,
        per_case_speedups=per_case_speedups,
    )
    observation = TriMulObservation(
        feedback=feedback, per_case_results=exec_results
    )
    logger.info(
        "TriMul Modal eval: reward={r}, elapsed={e:.1f}s, sha={sha}, "
        "outcome=success, n_cases={n}",
        r=f"{aggregated:.4f}",
        e=wall_clock_seconds,
        sha=sha[:8],
        n=len(exec_results),
    )
    record(sha, wall_clock_seconds, aggregated, len(exec_results), n_correct)
    return Evaluation[TriMulObservation](
        observation=observation, reward=aggregated
    )
