"""Async evaluation provider that fronts a surrogate then a real evaluator.

The compound provider implements
``AsyncEvaluationProvider[CompoundObservation[CaseSpeedupT]]``. Each
``submit(program_code)`` dispatches the abstaining surrogate; if the
surrogate forecasts, the outer Future resolves immediately with a
``ForecastObservation`` whose reward is materialized via the
``ForecastRewardPolicy``. If the surrogate defers, the candidate code
is forwarded to the real evaluator and the outer Future is chained to
the real evaluator's Future via ``add_done_callback``.

V2 async invariants honored:

* ``submit`` does not block. The outer Future is constructed up front
  and resolved from a coroutine running on an internal asyncio loop
  (forecast path) or from the inner provider's completion callback
  (deferral path).
* The internal asyncio loop runs in a daemon thread and is bounded
  by a semaphore on outbound surrogate calls.
* The real evaluator owns its own concurrency. Defer-path completions
  fire on whichever thread the inner provider runs its waiters on; we
  only call thread-safe ``Future.set_*`` on the outer.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future
from typing import Generic, Self

from loguru import logger

from gpu_forecasters.abstaining_evaluation.v1.forecast_reward import ForecastRewardPolicy
from gpu_forecasters.abstaining_evaluation.v1.observation import (
    CompoundObservation,
    ForecastObservation,
    RealObservation,
)
from gpu_forecasters.gpu_mode_kernel.core import (
    CaseSpeedupT,
    GpuModeKernelObservation,
)
from gpu_forecasters.hill_climbing.domain import Evaluation
from gpu_forecasters.landscape_map.v2 import (
    HardwareContext,
    KernelImplementation,
    KernelRuntimeEstimate,
    KernelRuntimeQuery,
    KernelTaskInfo,
)
from gpu_forecasters.landscape_map.v2.abstain_estimator import (
    AbstainingLlmSpeedupEstimator,
    Deferral,
    Forecast,
)
from gpu_forecasters.max_reward_puct.v2.providers import AsyncEvaluationProvider


class CompoundEvaluationProvider(Generic[CaseSpeedupT]):
    """Routes evaluations through a surrogate before the real GPU.

    Implements
    ``AsyncEvaluationProvider[CompoundObservation[CaseSpeedupT]]``.

    ``observation_type`` is required for the same reason the v2
    ``SearchDriver`` requires it: Pydantic generic ``BaseModel``s need
    the concrete subscription at construction time to serialize
    correctly. We hand it in once and use it whenever we build an
    ``Evaluation[CompoundObservation[...]]`` so the observation does
    not silently serialize as ``{}``.
    """

    def __init__(
        self,
        *,
        surrogate: AbstainingLlmSpeedupEstimator,
        real_evaluator: AsyncEvaluationProvider[GpuModeKernelObservation[CaseSpeedupT]],
        forecast_reward: ForecastRewardPolicy,
        task: KernelTaskInfo,
        reference: KernelImplementation,
        hardware: HardwareContext,
        observation_type: type[CompoundObservation[CaseSpeedupT]],
        candidate_kernel_name: str = "candidate",
        max_surrogate_concurrency: int = 8,
    ) -> None:
        if max_surrogate_concurrency < 1:
            raise ValueError("max_surrogate_concurrency must be >= 1")
        self._surrogate = surrogate
        self._real_evaluator = real_evaluator
        self._forecast_reward = forecast_reward
        self._task = task
        self._reference = reference
        self._hardware = hardware
        self._observation_type = observation_type
        self._candidate_kernel_name = candidate_kernel_name
        self._max_surrogate_concurrency = max_surrogate_concurrency

        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._loop_ready = threading.Event()

    # --- Lifecycle -----------------------------------------------------

    def __enter__(self) -> Self:
        # Enter the real evaluator first so a deferral-path submit can
        # always reach a ready inner provider.
        _ = self._real_evaluator.__enter__()
        self._loop_thread = threading.Thread(
            target=self._run_loop,
            name="abstaining-eval-loop",
            daemon=True,
        )
        self._loop_thread.start()
        _ = self._loop_ready.wait()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        # Stop our own loop first so no more submissions reach the
        # inner provider after we begin tearing it down.
        if self._loop is not None and self._loop.is_running():
            _ = self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=10.0)
        self._loop = None
        self._loop_thread = None
        self._semaphore = None
        self._loop_ready.clear()
        self._real_evaluator.__exit__(exc_type, exc_val, exc_tb)

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._semaphore = asyncio.Semaphore(self._max_surrogate_concurrency)
        self._loop_ready.set()
        try:
            loop.run_forever()
        finally:
            loop.close()

    # --- Submit --------------------------------------------------------

    def submit(
        self, program_code: str
    ) -> Future[Evaluation[CompoundObservation[CaseSpeedupT]]]:
        if self._loop is None or self._semaphore is None:
            raise RuntimeError(
                f"{type(self).__name__} must be entered as a context manager "
                "before submit()."
            )
        outer: Future[Evaluation[CompoundObservation[CaseSpeedupT]]] = Future()
        # Mark as running so set_result/set_exception transitions are
        # legal. The driver does not cancel evaluation futures, so the
        # branch where this returns False is dead in practice.
        if not outer.set_running_or_notify_cancel():
            return outer

        query = self._build_query(program_code)
        coro = self._run_compound(program_code, query, outer)
        _ = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return outer

    def _build_query(self, program_code: str) -> KernelRuntimeQuery:
        candidate = KernelImplementation(
            kernel_name=self._candidate_kernel_name,
            code=program_code,
            runtime_ms=None,
        )
        return KernelRuntimeQuery(
            task=self._task,
            reference=self._reference,
            candidate=candidate,
            hardware=self._hardware,
        )

    async def _run_compound(
        self,
        program_code: str,
        query: KernelRuntimeQuery,
        outer: Future[Evaluation[CompoundObservation[CaseSpeedupT]]],
    ) -> None:
        assert self._semaphore is not None
        try:
            async with self._semaphore:
                decision, _usage = await self._surrogate.aestimate(query)
        except BaseException as exc:
            logger.warning(
                "compound provider: surrogate call failed ({exc!r})", exc=exc
            )
            outer.set_exception(exc)
            return

        match decision:
            case Forecast(estimate=estimate):
                self._resolve_forecast(estimate, outer)
            case Deferral(reason=reason):
                self._dispatch_real_eval(program_code, reason, outer)

    def _resolve_forecast(
        self,
        estimate: KernelRuntimeEstimate,
        outer: Future[Evaluation[CompoundObservation[CaseSpeedupT]]],
    ) -> None:
        try:
            reward = self._forecast_reward(estimate)
            observation = ForecastObservation(
                estimate=estimate, expected_speedup=reward
            )
            evaluation: Evaluation[CompoundObservation[CaseSpeedupT]] = Evaluation[
                self._observation_type  # type: ignore[name-defined]
            ](observation=observation, reward=reward)
        except BaseException as exc:
            outer.set_exception(exc)
            return
        outer.set_result(evaluation)

    def _dispatch_real_eval(
        self,
        program_code: str,
        deferral_reason: str,
        outer: Future[Evaluation[CompoundObservation[CaseSpeedupT]]],
    ) -> None:
        try:
            inner_future = self._real_evaluator.submit(program_code)
        except BaseException as exc:
            outer.set_exception(exc)
            return
        # Callback runs on whichever thread the inner provider's waiter
        # completes on. Future.set_result/set_exception are thread-safe.
        inner_future.add_done_callback(
            lambda fut: self._propagate_inner(fut, deferral_reason, outer)
        )

    def _propagate_inner(
        self,
        inner_future: Future[Evaluation[GpuModeKernelObservation[CaseSpeedupT]]],
        deferral_reason: str,
        outer: Future[Evaluation[CompoundObservation[CaseSpeedupT]]],
    ) -> None:
        try:
            inner_eval = inner_future.result()
        except BaseException as exc:
            outer.set_exception(exc)
            return
        try:
            wrapped = RealObservation[CaseSpeedupT](
                inner=inner_eval.observation,
                deferral_reason=deferral_reason,
            )
            evaluation: Evaluation[CompoundObservation[CaseSpeedupT]] = Evaluation[
                self._observation_type  # type: ignore[name-defined]
            ](observation=wrapped, reward=inner_eval.reward)
        except BaseException as exc:
            outer.set_exception(exc)
            return
        outer.set_result(evaluation)


__all__ = ["CompoundEvaluationProvider"]
