"""Generic v2 Modal evaluation provider for gpu-mode-style kernels.

Generalizes ``arid_badger.max_reward_puct.v2.scoring_providers.{trimul,causal_conv1d}_modal``:
the asyncio-free thread-pool dispatch, app/cls lifecycle, and outcome
shaping are kernel-agnostic over a ``KernelPack``. The pack supplies
``case_speedup_factory``, the per-kernel Modal app, and the per-kernel
benchmarker cls (the latter two via constructor args because they are
declared at the pack module's import time and cannot live on the
frozen pack value).

Per-candidate contract: one ``submit(code)`` = exactly one remote
Modal call producing exactly one
``Evaluation[GpuModeKernelObservation[CaseSpeedupT]]``. Dispatch is
non-blocking via ``Function.spawn``; the caller gets a
``concurrent.futures.Future`` backed by an internal worker thread that
waits on ``FunctionCall.get()``.
"""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from types import TracebackType
from typing import Any, Generic, Literal, Optional, Self, cast

from loguru import logger
from pydantic import BaseModel

from arid_badger.gpu_mode_kernel.aggregation import (
    AggregationMethod,
    AggregationResult,
    aggregate_outcomes,
)
from arid_badger.gpu_mode_kernel.core import (
    CaseSpeedupT,
    GpuModeKernelObservation,
    InfrastructureFailureFeedback,
    KernelExecResult,
    RuntimeErrorFeedback,
    SuccessFeedback,
)
from arid_badger.gpu_mode_kernel.kernel_pack import TestArgsT
from arid_badger.gpu_mode_kernel.modal_scoring import PackedModalRuntime
from arid_badger.hill_climbing.domain import Evaluation
from arid_badger.invocation_sink import InvocationSink, code_sha256
from arid_badger.kernelbench.isolated_scoring import ScoringError
from arid_badger.typing_utils import Option


class GpuModeKernelEvaluationRecord(BaseModel, frozen=True):
    """Invocation record for a single gpu_mode_kernel Modal scoring call."""

    kind: Literal["gpu_mode_kernel_evaluation_v2"] = "gpu_mode_kernel_evaluation_v2"
    pack_name: str
    code_sha256: str
    wall_clock_seconds: float
    reward: float | None
    n_cases: int
    n_correct: int
    timestamp_utc: str


class GpuModeKernelModalProvider(Generic[TestArgsT, CaseSpeedupT]):
    """Async per-candidate evaluation provider for gpu-mode-style kernels.

    Implements ``AsyncEvaluationProvider[GpuModeKernelObservation[CaseSpeedupT]]``.

    Construction:
    - ``pack_runtime``: the pack + Modal app + benchmarker cls
      bundle, exported as a single constant by the pack's module.
    - ``test_cases``: the cases this provider scores on. Defaults to
      ``pack_runtime.pack.benchmark_cases`` if unset; experiments
      may pass a subset for calibration runs.
    """

    def __init__(
        self,
        *,
        pack_runtime: PackedModalRuntime[TestArgsT, CaseSpeedupT],
        test_cases: list[TestArgsT] | None = None,
        aggregator: AggregationMethod = "geomean",
        gpu: str = "A100-80GB",
        max_repeats: int = 100,
        max_time_ns: float = 10e9,
        max_in_flight: int = 10,
        max_containers: int = 10,
        get_timeout_s: float = 1200.0,
        invocation_sink: Optional[InvocationSink] = None,
    ) -> None:
        cases = (
            test_cases if test_cases is not None else pack_runtime.pack.benchmark_cases
        )
        if not cases:
            raise ValueError("test_cases must be non-empty")
        if max_in_flight < 1:
            raise ValueError("max_in_flight must be >= 1")
        if max_containers < 1:
            raise ValueError("max_containers must be >= 1")
        self._pack = pack_runtime.pack
        self._modal_app = pack_runtime.app
        self._benchmarker_cls = pack_runtime.benchmarker_cls
        self._test_cases = cases
        self._aggregator: AggregationMethod = aggregator
        self._gpu = gpu
        self._max_repeats = max_repeats
        self._max_time_ns = max_time_ns
        self._max_in_flight = max_in_flight
        self._max_containers = max_containers
        self._get_timeout_s = get_timeout_s
        self._invocation_sink = invocation_sink

        self._app_run_cm: AbstractContextManager[object] | None = None
        self._configured_benchmarker_cls: type | None = None
        self._waiter_pool: ThreadPoolExecutor | None = None

    # --- Lifecycle -----------------------------------------------------

    def __enter__(self) -> Self:
        self._app_run_cm = self._modal_app.run()
        _ = self._app_run_cm.__enter__()
        self._configured_benchmarker_cls = self._benchmarker_cls.with_options(  # type: ignore[attr-defined]
            gpu=self._gpu, max_containers=self._max_containers
        )
        self._waiter_pool = ThreadPoolExecutor(
            max_workers=self._max_in_flight,
            thread_name_prefix=f"{self._pack.name}-modal-waiter",
        )
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object,
    ) -> None:
        # Drain the waiter pool *before* tearing down the Modal app —
        # outstanding ``function_call.get()`` waits depend on the app
        # session being live.
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
        self._configured_benchmarker_cls = None

    # --- Submit --------------------------------------------------------

    def submit(
        self, program_code: str
    ) -> Future[Evaluation[GpuModeKernelObservation[CaseSpeedupT]]]:
        if self._configured_benchmarker_cls is None or self._waiter_pool is None:
            raise RuntimeError(
                f"{type(self).__name__} must be entered as a context manager "
                "before submit()."
            )
        start_time_s = time.perf_counter()
        function_call = self._configured_benchmarker_cls().evaluate_candidate.spawn(  # type: ignore[attr-defined]
            mutated_kernel_code=program_code,
            test_cases=[dict(cast(Any, tc)) for tc in self._test_cases],
            max_repeats=self._max_repeats,
            max_time_ns=self._max_time_ns,
        )
        return self._waiter_pool.submit(
            self._wait_and_wrap, program_code, function_call, start_time_s
        )

    def _wait_and_wrap(
        self,
        program_code: str,
        function_call: Any,
        start_time_s: float,
    ) -> Evaluation[GpuModeKernelObservation[CaseSpeedupT]]:
        sha = code_sha256(program_code)
        try:
            outcomes: list[Option[KernelExecResult, ScoringError]] = (
                function_call.get(timeout=self._get_timeout_s)
            )
        except Exception as exc:
            wall = time.perf_counter() - start_time_s
            logger.warning(
                "{name} Modal eval failed (sha={sha}, elapsed={e:.1f}s): {err}",
                name=self._pack.name,
                sha=sha[:8],
                e=wall,
                err=exc,
            )
            self._record(sha, wall, None, 0, 0)
            observation = GpuModeKernelObservation[CaseSpeedupT](
                feedback=InfrastructureFailureFeedback(
                    reason=f"modal call failed: {type(exc).__name__}: {exc}"
                ),
                per_case_results=[],
            )
            return Evaluation[GpuModeKernelObservation[CaseSpeedupT]](
                observation=observation, reward=None
            )

        wall = time.perf_counter() - start_time_s
        result = aggregate_outcomes(
            outcomes=outcomes,
            test_cases=self._test_cases,
            pack=self._pack,
            aggregator=self._aggregator,
        )
        observation = GpuModeKernelObservation[CaseSpeedupT](
            feedback=result.feedback,
            per_case_results=result.per_case_results,
        )
        self._record(
            sha, wall, result.reward, len(result.per_case_results), result.n_correct
        )
        self._log_outcome(result, sha=sha[:8], wall=wall)
        return Evaluation[GpuModeKernelObservation[CaseSpeedupT]](
            observation=observation, reward=result.reward
        )

    def _log_outcome(
        self,
        result: AggregationResult[CaseSpeedupT],
        *,
        sha: str,
        wall: float,
    ) -> None:
        feedback = result.feedback
        if isinstance(feedback, InfrastructureFailureFeedback):
            logger.warning(
                "{name} Modal eval: outcome=infra_failure, elapsed={e:.1f}s, "
                "sha={sha}, reason={reason}",
                name=self._pack.name,
                e=wall,
                sha=sha,
                reason=feedback.reason,
            )
            return
        if isinstance(feedback, SuccessFeedback):
            logger.info(
                "{name} Modal eval: reward={r:.4f}, elapsed={e:.1f}s, "
                "sha={sha}, outcome=success, n_cases={n}",
                name=self._pack.name,
                r=feedback.aggregated_speedup,
                e=wall,
                sha=sha,
                n=len(result.per_case_results),
            )
            return
        # Match legacy log shape: ``runtime_error({name})`` includes the
        # exception class name. Operators rely on this for grepping
        # specific failure modes out of run logs.
        outcome_label = (
            f"runtime_error({feedback.runtime_error_name})"
            if isinstance(feedback, RuntimeErrorFeedback)
            else feedback.kind
        )
        logger.info(
            "{name} Modal eval: outcome={kind}, elapsed={e:.1f}s, "
            "sha={sha}, n_correct={nc}/{n}",
            name=self._pack.name,
            kind=outcome_label,
            e=wall,
            sha=sha,
            nc=result.n_correct,
            n=len(result.per_case_results),
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
            GpuModeKernelEvaluationRecord(
                pack_name=self._pack.name,
                code_sha256=sha,
                wall_clock_seconds=wall,
                reward=reward,
                n_cases=n_cases,
                n_correct=n_correct,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
            )
        )
