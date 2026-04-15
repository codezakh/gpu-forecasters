"""Modal-based EvaluationProvider for TriMul kernels.

Parallel to ``kernelbench_modal.ModalProvider`` — same context-manager
lifecycle, same thread-pool ``batch_evaluate``, same invocation-sink
shape. Takes a ``test_args: TriMulTestArgs`` at construction (single
case per provider instance; multi-case aggregation is client-side).
"""

from __future__ import annotations

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
    CompileFailedFeedback,
    IncorrectFeedback,
    InfrastructureFailureFeedback,
    RuntimeErrorFeedback,
    SuccessFeedback,
    execution_feedback_from_exec_result,
)
from arid_badger.trimul.modal_scoring import (
    TriMulScoringFn,
    modal_trimul_scoring_session,
)
from arid_badger.typing_utils import implements, is_ok

from ..domain import Evaluation, EvaluationProvider
from .trimul import TriMulObservation


class ModalTriMulEvaluationRecord(BaseModel, frozen=True):
    """Invocation record for a single TriMul Modal scoring call."""

    kind: Literal["modal_trimul_evaluation"] = "modal_trimul_evaluation"
    code_sha256: str
    wall_clock_seconds: float
    reward: float | None
    timestamp_utc: str


class TriMulModalProvider:
    """Evaluates TriMul candidates on Modal.

    Must be used as a context manager to manage the Modal session lifecycle.
    """

    def __init__(
        self,
        test_args: TriMulTestArgs,
        gpu: str = "A100-80GB",
        max_repeats: int = 100,
        max_time_ns: float = 10e9,
        invocation_sink: Optional[InvocationSink] = None,
        max_batch_workers: int = 10,
    ) -> None:
        self._test_args = test_args
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

        sha = code_sha256(program_code)
        logger.info("TriMul Modal eval launching: sha256={sha}", sha=sha[:8])
        start_time_s = time.perf_counter()
        outcome = self._score_fn(program_code, self._test_args)
        wall_clock_seconds = time.perf_counter() - start_time_s

        if not is_ok(outcome):
            scoring_error = outcome.unwrap_err()
            logger.warning(
                "TriMul Modal eval failed after {elapsed:.1f}s: {reason}",
                elapsed=wall_clock_seconds,
                reason=scoring_error.reason,
            )
            observation = TriMulObservation(
                feedback=InfrastructureFailureFeedback(reason=scoring_error.reason),
            )
            if self._invocation_sink is not None:
                self._invocation_sink.record(
                    ModalTriMulEvaluationRecord(
                        code_sha256=sha,
                        wall_clock_seconds=wall_clock_seconds,
                        reward=None,
                        timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    )
                )
            return Evaluation[TriMulObservation](observation=observation, reward=None)

        exec_result = outcome.unwrap()
        speedup = (
            exec_result.ref_runtime_ns / exec_result.runtime_ns
            if exec_result.correct and exec_result.runtime_ns > 0
            else 0.0
        )
        feedback = execution_feedback_from_exec_result(
            exec_result=exec_result, speedup=speedup
        )
        observation = TriMulObservation(feedback=feedback)
        reward = speedup if exec_result.correct else None

        match feedback:
            case SuccessFeedback():
                outcome_name = "success"
            case CompileFailedFeedback():
                outcome_name = "compile_failed"
            case RuntimeErrorFeedback(runtime_error_name=name):
                outcome_name = f"runtime_error({name})"
            case IncorrectFeedback():
                outcome_name = "incorrect"
            case _:
                assert_never(feedback)

        logger.info(
            "TriMul Modal eval done: reward={reward}, elapsed={elapsed:.1f}s, "
            "sha256={sha}, outcome={outcome}",
            reward=f"{reward:.4f}" if reward is not None else "None",
            elapsed=wall_clock_seconds,
            sha=sha[:8],
            outcome=outcome_name,
        )

        if self._invocation_sink is not None:
            self._invocation_sink.record(
                ModalTriMulEvaluationRecord(
                    code_sha256=sha,
                    wall_clock_seconds=wall_clock_seconds,
                    reward=reward,
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                )
            )

        return Evaluation[TriMulObservation](observation=observation, reward=reward)

    def batch_evaluate(
        self, program_codes: list[str]
    ) -> list[Evaluation[TriMulObservation]]:
        if not program_codes:
            return []
        n_workers = min(len(program_codes), self._max_batch_workers)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            return list(executor.map(self.evaluate, program_codes))


implements(EvaluationProvider[TriMulObservation])(TriMulModalProvider)
