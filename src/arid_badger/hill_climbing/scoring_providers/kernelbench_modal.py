"""Modal-based EvaluationProvider for KernelBench kernels.

Drop-in alternative to `kernelbench.Provider` that runs compilation and
evaluation on Modal remote GPU containers rather than locally.

Because Modal requires an open session for remote calls, `ModalProvider`
must be used as a context manager. The session is opened on `__enter__`
and closed on `__exit__`:

    with ModalProvider(reference_kernel_code=ref_code, gpu="T4") as provider:
        best = search(..., evaluation_provider=provider)

This is the same pattern required by the `EvaluationProvider` protocol —
local providers implement no-op enter/exit, Modal manages its session.
"""

from __future__ import annotations

import logging
import time
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from types import TracebackType
from typing import Literal, Optional

from pydantic import BaseModel

from arid_badger.invocation_sink import InvocationSink, code_sha256
from arid_badger.kernelbench.core import (
    InfrastructureFailureFeedback,
    execution_feedback_from_exec_result,
)
from arid_badger.kernelbench.modal_scoring import modal_scoring_session, ScoringFn
from arid_badger.kernelbench.scoring import check_kernel_exec_result_valid
from arid_badger.typing_utils import is_ok
from ..domain import Evaluation, EvaluationProvider
from .kernelbench import KernelBenchObservation
from arid_badger.typing_utils import implements


class ModalEvaluationRecord(BaseModel, frozen=True):
    """Invocation record for a single Modal GPU kernel evaluation."""

    kind: Literal["modal_evaluation"] = "modal_evaluation"
    code_sha256: str
    wall_clock_seconds: float
    reward: float | None
    timestamp_utc: str


logger = logging.getLogger(__name__)


class ModalProvider(AbstractContextManager["ModalProvider"]):
    """Evaluates KernelBench kernels on Modal remote GPU containers.

    Must be used as a context manager to manage the Modal session lifecycle.
    See module docstring for usage.
    """

    def __init__(
        self,
        reference_kernel_code: str,
        gpu: str = "T4",
        backend: str = "cuda",
        precision: str = "fp32",
        num_correct_trials: int = 5,
        num_perf_trials: int = 100,
        invocation_sink: Optional[InvocationSink] = None,
    ) -> None:
        self._reference_kernel_code = reference_kernel_code
        self._gpu = gpu
        self._backend = backend
        self._precision = precision
        self._num_correct_trials = num_correct_trials
        self._num_perf_trials = num_perf_trials
        self._invocation_sink = invocation_sink
        self._session_cm: Optional[AbstractContextManager[ScoringFn]] = None
        self._score_fn: Optional[ScoringFn] = None

    def __enter__(self) -> ModalProvider:
        self._session_cm = modal_scoring_session(
            gpu=self._gpu,
            backend=self._backend,
            precision=self._precision,
            num_correct_trials=self._num_correct_trials,
            num_perf_trials=self._num_perf_trials,
        )
        self._score_fn = self._session_cm.__enter__()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> None:
        if self._session_cm is not None:
            self._session_cm.__exit__(exc_type, exc_val, exc_tb)

    def evaluate(self, program_code: str) -> Evaluation[KernelBenchObservation]:
        if self._score_fn is None:
            raise RuntimeError(
                "ModalProvider must be used as a context manager before calling evaluate(). "
                "Use: `with ModalProvider(...) as provider: provider.evaluate(...)`"
            )

        start_time_s = time.perf_counter()
        outcome = self._score_fn(
            program_code,
            self._reference_kernel_code,
        )
        wall_clock_seconds = time.perf_counter() - start_time_s

        if not is_ok(outcome):
            scoring_error = outcome.unwrap_err()
            logger.warning("Modal kernel scoring failed: %s", scoring_error.reason)
            observation = KernelBenchObservation(
                feedback=InfrastructureFailureFeedback(reason=scoring_error.reason),
            )
            if self._invocation_sink is not None:
                self._invocation_sink.record(
                    ModalEvaluationRecord(
                        code_sha256=code_sha256(program_code),
                        wall_clock_seconds=wall_clock_seconds,
                        reward=None,
                        timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    )
                )
            return Evaluation[KernelBenchObservation](
                observation=observation, reward=None
            )

        exec_result = outcome.unwrap()
        is_valid = check_kernel_exec_result_valid(exec_result)
        speedup = exec_result.ref_runtime / exec_result.runtime if is_valid else 0.0

        feedback = execution_feedback_from_exec_result(
            exec_result=exec_result,
            speedup=speedup,
            is_valid=is_valid,
        )
        observation = KernelBenchObservation(feedback=feedback)
        reward = speedup if is_valid else None

        if self._invocation_sink is not None:
            self._invocation_sink.record(
                ModalEvaluationRecord(
                    code_sha256=code_sha256(program_code),
                    wall_clock_seconds=wall_clock_seconds,
                    reward=reward,
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                )
            )

        return Evaluation[KernelBenchObservation](
            observation=observation,
            reward=reward,
        )


implements(EvaluationProvider[KernelBenchObservation])(ModalProvider)
