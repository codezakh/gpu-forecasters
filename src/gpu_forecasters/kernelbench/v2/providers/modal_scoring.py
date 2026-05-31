"""V2 async evaluation provider for KernelBench kernels.

Async-chained over the split CPU-compile / GPU-benchmark Modal pipeline
(``gpu_forecasters.kernelbench.modal_split_scoring``, ADR-002). One submit
== one in-flight unit at any moment: submit returns a
``concurrent.futures.Future`` immediately; the underlying coroutine
awaits two ``.remote.aio(...)`` calls (compile, then bench) chained on a
single asyncio loop running on a dedicated background thread.

Why not wrap the v1 sync ``score()`` in a thread-pool executor? That
would defeat the v2 contract — ``score()`` chains two *blocking*
``.remote()`` calls, so each in-flight candidate would tie up an OS
thread for the full compile+bench duration. The v2 driver would see
futures that look in-flight but are actually parked on synchronous
network calls. With ``.remote.aio(...)`` chained inside a coroutine,
outstanding candidates park on ``await`` points and are multiplexed on
one OS thread by the asyncio loop — the split pipeline's two-stage
container topology is preserved exactly, just expressed as a coroutine
on the client.

Mirrors the lifecycle pattern in
``gpu_forecasters.gpu_mode_kernel.providers.v2_feedback_mutation``.
"""

from __future__ import annotations

import asyncio
import threading
import time
import traceback
from concurrent.futures import Future
from contextlib import AbstractContextManager
from datetime import datetime, timezone
from types import TracebackType
from typing import Any, Literal, Self, cast

from loguru import logger
from pydantic import BaseModel

from gpu_forecasters.hill_climbing.domain import Evaluation
from gpu_forecasters.hill_climbing.scoring_providers.kernelbench import (
    KernelBenchObservation,
)
from gpu_forecasters.invocation_sink import InvocationSink, code_sha256
from gpu_forecasters.kernelbench.core import (
    InfrastructureFailureFeedback,
    execution_feedback_from_exec_result,
)
from gpu_forecasters.kernelbench.modal_image import GPU_ARCH_MAPPING
from gpu_forecasters.kernelbench.modal_split_scoring import (
    COMPUTE_CAPABILITY_BY_GPU,
    ModalCpuCompiler,
    ModalGpuBenchmarker,
    app,
)
from gpu_forecasters.kernelbench.scoring import check_kernel_exec_result_valid
from gpu_forecasters.max_reward_puct.v2.providers import AsyncEvaluationProvider
from gpu_forecasters.modal_gpu import GpuKind
from gpu_forecasters.typing_utils import implements
from kernelbench.eval import KernelExecResult


class KernelBenchModalEvaluationRecord(BaseModel, frozen=True):
    """Invocation record for a single v2 KernelBench Modal evaluation.

    Distinct ``kind`` from the v1 ``ModalEvaluationRecord`` so a run that
    happens to mix v1 and v2 providers (or a v1 run replayed under v2
    tooling) keeps the two streams separable on disk.
    """

    kind: Literal["kernelbench_modal_evaluation_v2"] = (
        "kernelbench_modal_evaluation_v2"
    )
    code_sha256: str
    wall_clock_seconds: float
    reward: float | None
    timestamp_utc: str


class KernelBenchModalProvider:
    """Per-candidate async evaluation provider for KernelBench kernels.

    Implements ``AsyncEvaluationProvider[KernelBenchObservation]``.

    Lifecycle:
        ``__enter__`` opens ``app.run()`` on the existing
        ``arid-badger-kernel-split`` Modal app, binds
        ``ModalCpuCompiler`` / ``ModalGpuBenchmarker.with_options(gpu=...)``
        handles, and starts a dedicated asyncio loop on a background
        thread. ``__exit__`` stops the loop, joins the thread, and
        closes the app session.

    Per-candidate contract:
        ``submit(code)`` returns immediately with a
        ``Future[Evaluation[KernelBenchObservation]]``. The coroutine
        backing that future does ``compile.remote.aio`` →
        ``evaluate.remote.aio``, holding an ``asyncio.Semaphore`` slot
        for the full duration so ``max_in_flight`` bounds genuine
        concurrency (not just submitted-but-queued work).

    Failure handling:
        Both Modal stages can raise (network, container crash, etc.).
        These are caught inside the coroutine and surfaced as
        ``Evaluation`` with ``InfrastructureFailureFeedback`` — matching
        the v1 ``ModalProvider`` shape so the v2 driver only ever sees
        successful futures whose payloads encode the failure mode.
        ``compile_result["error"]`` (CPU compile error in the user
        kernel) becomes a synthesized ``KernelExecResult(compiled=False,
        ...)`` routed through the same ``execution_feedback_from_exec_result``
        path the v1 provider uses, so the resulting feedback type is
        ``CompileFailedFeedback``, not ``InfrastructureFailureFeedback``.
    """

    def __init__(
        self,
        *,
        reference_kernel_code: str,
        gpu: GpuKind = GpuKind.L4,
        backend: str = "cuda",
        precision: str = "fp32",
        num_correct_trials: int = 5,
        num_perf_trials: int = 100,
        max_in_flight: int = 8,
        invocation_sink: InvocationSink | None = None,
    ) -> None:
        if max_in_flight < 1:
            raise ValueError("max_in_flight must be >= 1")
        if gpu.value not in COMPUTE_CAPABILITY_BY_GPU:
            raise ValueError(
                f"GPU {gpu.value!r} is in GpuKind but missing from "
                f"COMPUTE_CAPABILITY_BY_GPU (gpu_forecasters.kernelbench."
                f"modal_split_scoring). Add the entry before using "
                f"KernelBenchModalProvider."
            )
        self._reference_kernel_code = reference_kernel_code
        self._gpu = gpu
        self._cc = COMPUTE_CAPABILITY_BY_GPU[gpu.value]
        self._gpu_arch = GPU_ARCH_MAPPING.get(gpu.value, ["Ampere"])
        self._backend = backend
        self._precision = precision
        self._num_correct_trials = num_correct_trials
        self._num_perf_trials = num_perf_trials
        self._max_in_flight = max_in_flight
        self._invocation_sink = invocation_sink

        # Lifecycle-owned state. All None until ``__enter__``.
        self._app_session: AbstractContextManager[Any] | None = None
        self._compiler: Any | None = None
        self._benchmarker_cls: Any | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._loop_ready = threading.Event()

    # --- Lifecycle ------------------------------------------------------

    def __enter__(self) -> Self:
        # Open the Modal session synchronously so any failure surfaces
        # at ``with`` entry rather than buried inside a future.
        self._app_session = app.run()
        self._app_session.__enter__()
        try:
            self._compiler = ModalCpuCompiler()
            self._benchmarker_cls = ModalGpuBenchmarker.with_options(  # ty: ignore[unresolved-attribute]  # pyright: ignore[reportAttributeAccessIssue]
                gpu=self._gpu
            )

            self._loop_thread = threading.Thread(
                target=self._run_loop,
                name="kernelbench-modal-provider-loop",
                daemon=True,
            )
            self._loop_thread.start()
            self._loop_ready.wait()
        except BaseException:
            # Roll back the partial open so the app session does not
            # leak if anything between here and a successful return
            # raises.
            self._app_session.__exit__(None, None, None)
            self._app_session = None
            raise
        return self

    def __exit__(
        self,
        exc_type: object,
        exc_val: object,
        exc_tb: object,
    ) -> None:
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=10.0)
        if self._app_session is not None:
            self._app_session.__exit__(
                cast(type[BaseException] | None, exc_type),
                cast(BaseException | None, exc_val),
                cast(TracebackType | None, exc_tb),
            )
        self._app_session = None
        self._compiler = None
        self._benchmarker_cls = None
        self._loop = None
        self._loop_thread = None
        self._semaphore = None
        self._loop_ready.clear()

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        # The semaphore must be constructed on the loop that will
        # acquire it — asyncio synchronization primitives bind to the
        # running loop on first await.
        self._semaphore = asyncio.Semaphore(self._max_in_flight)
        self._loop_ready.set()
        try:
            loop.run_forever()
        finally:
            loop.close()

    # --- Submit ---------------------------------------------------------

    def submit(
        self, program_code: str
    ) -> Future[Evaluation[KernelBenchObservation]]:
        if (
            self._loop is None
            or self._semaphore is None
            or self._compiler is None
            or self._benchmarker_cls is None
        ):
            raise RuntimeError(
                f"{type(self).__name__} must be entered as a context manager "
                "before submit()."
            )
        coro = self._score_one_async(program_code)
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    async def _score_one_async(
        self, code: str
    ) -> Evaluation[KernelBenchObservation]:
        assert self._semaphore is not None
        assert self._compiler is not None
        assert self._benchmarker_cls is not None

        sha = code_sha256(code)
        start_time_s = time.perf_counter()

        async with self._semaphore:
            try:
                compile_result: dict[str, Any] = (
                    await self._compiler.compile.remote.aio(
                        code,
                        self._reference_kernel_code,
                        self._cc,
                    )
                )
            except Exception as exc:
                logger.warning(
                    "Modal CPU compile call failed: {exc}\n{tb}",
                    exc=exc,
                    tb=traceback.format_exc(),
                )
                return self._infrastructure_failure(
                    f"Modal CPU compile failed: {type(exc).__name__}: {exc}",
                    sha=sha,
                    start_time_s=start_time_s,
                )

            cpu_error = compile_result.get("error")
            if cpu_error:
                # User kernel failed nvcc on the CPU container — this is
                # a kernel defect, not infrastructure. Route through the
                # same ``execution_feedback_from_exec_result`` path the
                # v1 provider uses so it surfaces as
                # ``CompileFailedFeedback``.
                return self._evaluation_from_exec_result(
                    KernelExecResult(
                        compiled=False,
                        metadata={
                            "compilation_error_name": "CpuCompileError",
                            "compilation_error": cpu_error,
                        },
                    ),
                    sha=sha,
                    start_time_s=start_time_s,
                )

            cache_dir = compile_result["cache_dir"]
            try:
                exec_result: KernelExecResult | None = (
                    await self._benchmarker_cls().evaluate.remote.aio(
                        mutated_kernel_code=code,
                        reference_kernel_code=self._reference_kernel_code,
                        cache_dir=cache_dir,
                        gpu_arch=self._gpu_arch,
                        backend=self._backend,
                        precision=self._precision,
                        num_correct_trials=self._num_correct_trials,
                        num_perf_trials=self._num_perf_trials,
                    )
                )
            except Exception as exc:
                logger.warning(
                    "Modal GPU benchmark call failed: {exc}\n{tb}",
                    exc=exc,
                    tb=traceback.format_exc(),
                )
                return self._infrastructure_failure(
                    f"Modal GPU benchmark failed: {type(exc).__name__}: {exc}",
                    sha=sha,
                    start_time_s=start_time_s,
                )

            if exec_result is None:
                # ``eval_kernel_against_ref`` returns None on lock-file /
                # "No such file or directory" races during compilation.
                # The LLM cannot act on this feedback — it cannot fix a
                # file-system race — so route to InfrastructureFailureFeedback
                # rather than a misleading "compile failed" prompt.
                return self._infrastructure_failure(
                    "eval_kernel_against_ref returned None "
                    "(likely lock-file race during compilation)",
                    sha=sha,
                    start_time_s=start_time_s,
                )

            return self._evaluation_from_exec_result(
                exec_result, sha=sha, start_time_s=start_time_s
            )

    # --- Wrapping helpers ----------------------------------------------

    def _infrastructure_failure(
        self,
        reason: str,
        *,
        sha: str,
        start_time_s: float,
    ) -> Evaluation[KernelBenchObservation]:
        wall = time.perf_counter() - start_time_s
        self._record(sha=sha, wall_clock_seconds=wall, reward=None)
        return Evaluation[KernelBenchObservation](
            observation=KernelBenchObservation(
                feedback=InfrastructureFailureFeedback(reason=reason),
            ),
            reward=None,
        )

    def _evaluation_from_exec_result(
        self,
        exec_result: KernelExecResult,
        *,
        sha: str,
        start_time_s: float,
    ) -> Evaluation[KernelBenchObservation]:
        is_valid = check_kernel_exec_result_valid(exec_result)
        speedup = (
            exec_result.ref_runtime / exec_result.runtime if is_valid else 0.0
        )
        feedback = execution_feedback_from_exec_result(
            exec_result=exec_result,
            speedup=speedup,
            is_valid=is_valid,
        )
        reward = speedup if is_valid else None
        wall = time.perf_counter() - start_time_s
        self._record(sha=sha, wall_clock_seconds=wall, reward=reward)
        return Evaluation[KernelBenchObservation](
            observation=KernelBenchObservation(feedback=feedback),
            reward=reward,
        )

    # --- Sink bookkeeping ---------------------------------------------

    def _record(
        self,
        *,
        sha: str,
        wall_clock_seconds: float,
        reward: float | None,
    ) -> None:
        if self._invocation_sink is None:
            return
        self._invocation_sink.record(
            KernelBenchModalEvaluationRecord(
                code_sha256=sha,
                wall_clock_seconds=wall_clock_seconds,
                reward=reward,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
            )
        )


implements(AsyncEvaluationProvider[KernelBenchObservation])(
    KernelBenchModalProvider
)
