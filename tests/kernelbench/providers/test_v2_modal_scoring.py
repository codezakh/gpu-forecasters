"""Unit tests for the v2 KernelBench Modal evaluation provider.

These tests mock the Modal class handles so they exercise the asyncio
plumbing (loop-on-thread + semaphore + chained awaits + failure
wrapping) without needing Modal authentication. The end-to-end
integration test that hits real Modal lives in
``test_v2_modal_scoring_integration.py`` and is gated behind
``@pytest.mark.modal``.
"""

from __future__ import annotations

import asyncio
import threading
from contextlib import contextmanager
from typing import Any, Callable, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from kernelbench.eval import KernelExecResult

from arid_badger.kernelbench.core import (
    CompileFailedFeedback,
    IncorrectFeedback,
    InfrastructureFailureFeedback,
    SuccessFeedback,
)
from arid_badger.invocation_sink import code_sha256
from arid_badger.kernelbench.providers.v2_modal_scoring import (
    KernelBenchModalEvaluationRecord,
    KernelBenchModalProvider,
)
from pydantic import BaseModel

PROVIDER_MODULE = "arid_badger.kernelbench.providers.v2_modal_scoring"

_REFERENCE_KERNEL = "reference-kernel-source"
_CANDIDATE_KERNEL = "candidate-kernel-source"


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _mock_app_session() -> MagicMock:
    """Stand-in for ``app.run()`` returning a no-op context manager."""
    session = MagicMock()
    session.__enter__.return_value = None
    session.__exit__.return_value = None
    app = MagicMock()
    app.run.return_value = session
    return app


def _ok_exec_result(*, runtime: float = 10.0, ref_runtime: float = 20.0) -> KernelExecResult:
    return KernelExecResult(
        compiled=True,
        correctness=True,
        runtime=runtime,
        ref_runtime=ref_runtime,
        metadata={},
    )


def _incorrect_exec_result() -> KernelExecResult:
    return KernelExecResult(
        compiled=True,
        correctness=False,
        runtime=10.0,
        ref_runtime=20.0,
        metadata={
            "correctness_issue": "outputs disagree",
            "max_difference": [0.5],
            "avg_difference": [0.1],
        },
    )


class _ListSink:
    """List-backed ``InvocationSink`` for tests — captures records in
    insertion order so a test can assert on counts and field values."""

    def __init__(self) -> None:
        self.records: list[BaseModel] = []

    def record(self, payload: BaseModel) -> None:
        self.records.append(payload)


@contextmanager
def _patched_provider(
    *,
    compile_side_effect: Callable[..., Any] | None = None,
    compile_return: Any | None = None,
    evaluate_side_effect: Callable[..., Any] | None = None,
    evaluate_return: Any | None = None,
    max_in_flight: int = 8,
    gpu: str = "L4",
    invocation_sink: _ListSink | None = None,
) -> Generator[KernelBenchModalProvider, None, None]:
    """Context manager that yields a KernelBenchModalProvider whose Modal
    handles are mocked.

    Exactly one of ``compile_side_effect`` / ``compile_return`` should be
    set per call; same for evaluate. ``compile_return`` is a convenience
    that wires up a static return value.
    """
    fake_app = _mock_app_session()
    fake_compiler = MagicMock()
    fake_compiler.compile.remote.aio = AsyncMock(
        side_effect=compile_side_effect,
        return_value=compile_return,
    )
    fake_benchmarker_instance = MagicMock()
    fake_benchmarker_instance.evaluate.remote.aio = AsyncMock(
        side_effect=evaluate_side_effect,
        return_value=evaluate_return,
    )
    # ``self._benchmarker_cls()`` instantiates a benchmarker; we want each
    # call to return the same mocked instance so its ``evaluate`` mock is
    # observable to the test.
    fake_benchmarker_cls = MagicMock(return_value=fake_benchmarker_instance)
    fake_benchmarker_class = MagicMock()
    fake_benchmarker_class.with_options.return_value = fake_benchmarker_cls

    with (
        patch(f"{PROVIDER_MODULE}.app", fake_app),
        patch(f"{PROVIDER_MODULE}.ModalCpuCompiler", return_value=fake_compiler),
        patch(f"{PROVIDER_MODULE}.ModalGpuBenchmarker", fake_benchmarker_class),
    ):
        provider = KernelBenchModalProvider(
            reference_kernel_code=_REFERENCE_KERNEL,
            gpu=gpu,
            max_in_flight=max_in_flight,
            invocation_sink=invocation_sink,
        )
        with provider:
            yield provider


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def test_lifecycle_starts_and_stops_loop_thread() -> None:
    """``__enter__`` boots the asyncio loop on a background thread and
    ``__exit__`` joins it cleanly."""
    with _patched_provider(
        compile_return={"cache_dir": "/cache/x", "error": None},
        evaluate_return=_ok_exec_result(),
    ) as provider:
        assert provider._loop_thread is not None
        assert provider._loop_thread.is_alive()
        assert provider._loop is not None

    assert provider._loop is None
    assert provider._loop_thread is None
    assert provider._semaphore is None


def test_submit_before_enter_raises() -> None:
    """``submit()`` outside the context manager is an error — there is no
    loop to dispatch onto."""
    fake_app = _mock_app_session()
    with patch(f"{PROVIDER_MODULE}.app", fake_app):
        provider = KernelBenchModalProvider(
            reference_kernel_code=_REFERENCE_KERNEL,
            gpu="L4",
        )
        with pytest.raises(RuntimeError, match="must be entered as a context manager"):
            provider.submit(_CANDIDATE_KERNEL)


def test_invalid_max_in_flight_raises() -> None:
    """``max_in_flight < 1`` is incoherent (no work could ever start)."""
    with pytest.raises(ValueError, match="max_in_flight must be >= 1"):
        KernelBenchModalProvider(
            reference_kernel_code=_REFERENCE_KERNEL,
            gpu="L4",
            max_in_flight=0,
        )


def test_unknown_gpu_raises() -> None:
    """A GPU not in COMPUTE_CAPABILITY_BY_GPU has no compute capability
    string to pass to nvcc — must fail at construction."""
    with pytest.raises(ValueError, match="Unknown GPU"):
        KernelBenchModalProvider(
            reference_kernel_code=_REFERENCE_KERNEL,
            gpu="not-a-real-gpu",
        )


# ---------------------------------------------------------------------------
# Submit: success / failure shapes
# ---------------------------------------------------------------------------


def test_submit_success_returns_evaluation_with_speedup() -> None:
    """Happy path: compile clean, exec_result correct, reward equals
    ``ref_runtime / runtime`` (the v1 ModalProvider's speedup convention)."""
    with _patched_provider(
        compile_return={"cache_dir": "/cache/x", "error": None},
        evaluate_return=_ok_exec_result(runtime=5.0, ref_runtime=20.0),
    ) as provider:
        future = provider.submit(_CANDIDATE_KERNEL)
        evaluation = future.result(timeout=5.0)

    assert evaluation.reward == pytest.approx(4.0)
    assert isinstance(evaluation.observation.feedback, SuccessFeedback)
    assert evaluation.observation.feedback.speedup == pytest.approx(4.0)
    assert evaluation.observation.feedback.runtime_us == 5.0
    assert evaluation.observation.feedback.ref_runtime_us == 20.0


def test_submit_compile_failure_routes_to_compile_failed_feedback() -> None:
    """``compile_result["error"]`` (CPU compile rejected the user kernel)
    must surface as ``CompileFailedFeedback`` — not infrastructure
    failure — so the LLM sees a kernel-defect signal it can act on."""
    with _patched_provider(
        compile_return={
            "cache_dir": "/cache/x",
            "error": "nvcc: error: unknown arg",
        },
        # evaluate must never be called when compile errored
        evaluate_return=None,
    ) as provider:
        future = provider.submit(_CANDIDATE_KERNEL)
        evaluation = future.result(timeout=5.0)

    assert evaluation.reward is None
    assert isinstance(evaluation.observation.feedback, CompileFailedFeedback)
    assert evaluation.observation.feedback.compilation_error_name == "CpuCompileError"
    assert "nvcc" in evaluation.observation.feedback.compilation_error


def test_submit_compile_call_raises_routes_to_infrastructure_failure() -> None:
    """A raised exception inside the compile coroutine (network drop,
    Modal infra error, etc.) becomes
    ``InfrastructureFailureFeedback`` — not a per-kernel signal."""

    async def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("modal connection lost")

    with _patched_provider(compile_side_effect=boom) as provider:
        future = provider.submit(_CANDIDATE_KERNEL)
        evaluation = future.result(timeout=5.0)

    assert evaluation.reward is None
    assert isinstance(
        evaluation.observation.feedback, InfrastructureFailureFeedback
    )
    assert "modal connection lost" in evaluation.observation.feedback.reason


def test_submit_evaluate_call_raises_routes_to_infrastructure_failure() -> None:
    """Same shape as the compile-call-raises case but at the GPU
    benchmark stage — must still surface as
    ``InfrastructureFailureFeedback`` with ``reward=None``."""

    async def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("gpu container crashed")

    with _patched_provider(
        compile_return={"cache_dir": "/cache/x", "error": None},
        evaluate_side_effect=boom,
    ) as provider:
        future = provider.submit(_CANDIDATE_KERNEL)
        evaluation = future.result(timeout=5.0)

    assert evaluation.reward is None
    assert isinstance(
        evaluation.observation.feedback, InfrastructureFailureFeedback
    )
    assert "gpu container crashed" in evaluation.observation.feedback.reason


def test_submit_evaluate_returns_none_routes_to_infrastructure_failure() -> None:
    """``eval_kernel_against_ref`` returns None on lock-file races. The
    LLM cannot act on this; route to infrastructure failure rather than
    a misleading 'compile failed' prompt."""
    with _patched_provider(
        compile_return={"cache_dir": "/cache/x", "error": None},
        evaluate_return=None,
    ) as provider:
        future = provider.submit(_CANDIDATE_KERNEL)
        evaluation = future.result(timeout=5.0)

    assert evaluation.reward is None
    assert isinstance(
        evaluation.observation.feedback, InfrastructureFailureFeedback
    )
    assert "lock-file" in evaluation.observation.feedback.reason


def test_submit_incorrect_kernel_returns_incorrect_feedback() -> None:
    """``correctness=False`` (with no runtime_error metadata) becomes
    ``IncorrectFeedback`` and ``reward=None``."""
    with _patched_provider(
        compile_return={"cache_dir": "/cache/x", "error": None},
        evaluate_return=_incorrect_exec_result(),
    ) as provider:
        future = provider.submit(_CANDIDATE_KERNEL)
        evaluation = future.result(timeout=5.0)

    assert evaluation.reward is None
    assert isinstance(evaluation.observation.feedback, IncorrectFeedback)
    assert evaluation.observation.feedback.correctness_issue == "outputs disagree"


# ---------------------------------------------------------------------------
# Concurrency: semaphore enforces max_in_flight
# ---------------------------------------------------------------------------


def test_max_in_flight_semaphore_caps_concurrent_modal_calls() -> None:
    """Submitting more kernels than ``max_in_flight`` must not run more
    than ``max_in_flight`` Modal stages in parallel.

    Strategy:
        Replace ``compile.remote.aio`` with a coroutine that increments
        an in-flight counter, parks on an asyncio.Event, then decrements
        on release. With ``max_in_flight=2`` and 5 submitted candidates,
        the recorded peak must equal 2.
    """
    max_in_flight = 2
    n_submits = 5

    in_flight = 0
    peak_in_flight = 0
    counter_lock = threading.Lock()

    async def slow_compile(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal in_flight, peak_in_flight
        with counter_lock:
            in_flight += 1
            peak_in_flight = max(peak_in_flight, in_flight)
        try:
            await release_event.wait()
        finally:
            with counter_lock:
                in_flight -= 1
        return {"cache_dir": "/cache/x", "error": None}

    fake_app = _mock_app_session()
    fake_compiler = MagicMock()
    fake_compiler.compile.remote.aio = AsyncMock(side_effect=slow_compile)
    fake_benchmarker_instance = MagicMock()
    fake_benchmarker_instance.evaluate.remote.aio = AsyncMock(
        return_value=_ok_exec_result()
    )
    fake_benchmarker_cls = MagicMock(return_value=fake_benchmarker_instance)
    fake_benchmarker_class = MagicMock()
    fake_benchmarker_class.with_options.return_value = fake_benchmarker_cls

    with (
        patch(f"{PROVIDER_MODULE}.app", fake_app),
        patch(f"{PROVIDER_MODULE}.ModalCpuCompiler", return_value=fake_compiler),
        patch(f"{PROVIDER_MODULE}.ModalGpuBenchmarker", fake_benchmarker_class),
    ):
        with KernelBenchModalProvider(
            reference_kernel_code=_REFERENCE_KERNEL,
            gpu="L4",
            max_in_flight=max_in_flight,
        ) as provider:
            assert provider._loop is not None

            # Allocate the asyncio.Event on the provider's loop so it
            # binds to the right loop on first await.
            async def _make_event() -> asyncio.Event:
                return asyncio.Event()

            release_event = asyncio.run_coroutine_threadsafe(
                _make_event(), provider._loop
            ).result(timeout=5.0)

            futures = [provider.submit(_CANDIDATE_KERNEL) for _ in range(n_submits)]

            # Give the loop time to schedule as many coroutines as the
            # semaphore allows; they then park on ``release_event``.
            _wait_for(lambda: in_flight >= max_in_flight, timeout=5.0)

            # Brief pause to confirm no further coroutines slip past the
            # semaphore. ``asyncio.run_coroutine_threadsafe`` returns
            # immediately, so any extra coroutines would be visible here.
            _spin(0.2)

            with counter_lock:
                observed_peak = peak_in_flight

            # Release all parked coroutines so the futures can complete.
            provider._loop.call_soon_threadsafe(release_event.set)

            evaluations = [f.result(timeout=10.0) for f in futures]

    assert observed_peak == max_in_flight, (
        f"Expected peak concurrency {max_in_flight}, observed {observed_peak}. "
        f"Semaphore is not bounding in-flight Modal calls."
    )
    assert len(evaluations) == n_submits
    for ev in evaluations:
        assert isinstance(ev.observation.feedback, SuccessFeedback)


def _wait_for(predicate: Callable[[], bool], *, timeout: float) -> None:
    """Spin until ``predicate()`` is True or ``timeout`` elapses."""
    import time

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(f"predicate never became True within {timeout}s")


def _spin(duration: float) -> None:
    import time

    time.sleep(duration)


# ---------------------------------------------------------------------------
# Concurrency: independent submits don't serialize through one OS thread
# ---------------------------------------------------------------------------


def test_concurrent_submits_resolve_independently() -> None:
    """A slow candidate must not block faster ones from resolving.

    Setup:
        Two submits. The first parks on an event for the entire test;
        the second returns immediately. The second's future must
        resolve while the first is still parked — proving the asyncio
        loop is multiplexing, not serializing.
    """
    fake_app = _mock_app_session()

    submit_count = 0

    async def compile_fn(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal submit_count
        submit_count += 1
        # First submit parks; second returns immediately.
        if submit_count == 1:
            await first_park_event.wait()
        return {"cache_dir": "/cache/x", "error": None}

    fake_compiler = MagicMock()
    fake_compiler.compile.remote.aio = AsyncMock(side_effect=compile_fn)
    fake_benchmarker_instance = MagicMock()
    fake_benchmarker_instance.evaluate.remote.aio = AsyncMock(
        return_value=_ok_exec_result()
    )
    fake_benchmarker_cls = MagicMock(return_value=fake_benchmarker_instance)
    fake_benchmarker_class = MagicMock()
    fake_benchmarker_class.with_options.return_value = fake_benchmarker_cls

    with (
        patch(f"{PROVIDER_MODULE}.app", fake_app),
        patch(f"{PROVIDER_MODULE}.ModalCpuCompiler", return_value=fake_compiler),
        patch(f"{PROVIDER_MODULE}.ModalGpuBenchmarker", fake_benchmarker_class),
    ):
        with KernelBenchModalProvider(
            reference_kernel_code=_REFERENCE_KERNEL,
            gpu="L4",
            max_in_flight=4,
        ) as provider:
            assert provider._loop is not None

            async def _make_event() -> asyncio.Event:
                return asyncio.Event()

            first_park_event = asyncio.run_coroutine_threadsafe(
                _make_event(), provider._loop
            ).result(timeout=5.0)

            slow_future = provider.submit(_CANDIDATE_KERNEL)
            fast_future = provider.submit(_CANDIDATE_KERNEL)

            # Fast future completes while slow is still parked.
            fast_eval = fast_future.result(timeout=5.0)
            assert isinstance(fast_eval.observation.feedback, SuccessFeedback)
            assert not slow_future.done()

            # Release slow.
            provider._loop.call_soon_threadsafe(first_park_event.set)
            slow_eval = slow_future.result(timeout=5.0)
            assert isinstance(slow_eval.observation.feedback, SuccessFeedback)


# ---------------------------------------------------------------------------
# Invocation sink — every terminal path must produce exactly one record
# ---------------------------------------------------------------------------


def test_sink_records_success() -> None:
    """Happy path: one record, ``reward`` matches the speedup, ``code_sha256``
    matches the submitted code."""
    sink = _ListSink()
    with _patched_provider(
        compile_return={"cache_dir": "/cache/x", "error": None},
        evaluate_return=_ok_exec_result(runtime=5.0, ref_runtime=20.0),
        invocation_sink=sink,
    ) as provider:
        provider.submit(_CANDIDATE_KERNEL).result(timeout=5.0)

    assert len(sink.records) == 1
    record = sink.records[0]
    assert isinstance(record, KernelBenchModalEvaluationRecord)
    assert record.code_sha256 == code_sha256(_CANDIDATE_KERNEL)
    assert record.reward == pytest.approx(4.0)
    assert record.wall_clock_seconds >= 0.0


def test_sink_records_compile_failure_with_reward_none() -> None:
    """CPU compile failure surfaces as ``CompileFailedFeedback`` for the
    LLM, but the sink record carries ``reward=None`` — a kernel defect
    contributed no usable reward signal."""
    sink = _ListSink()
    with _patched_provider(
        compile_return={
            "cache_dir": "/cache/x",
            "error": "nvcc: error: unknown arg",
        },
        invocation_sink=sink,
    ) as provider:
        provider.submit(_CANDIDATE_KERNEL).result(timeout=5.0)

    assert len(sink.records) == 1
    record = sink.records[0]
    assert isinstance(record, KernelBenchModalEvaluationRecord)
    assert record.reward is None
    assert record.code_sha256 == code_sha256(_CANDIDATE_KERNEL)


def test_sink_records_infrastructure_failure_when_compile_call_raises() -> None:
    """Modal connection drop (compile-stage) → one record with reward=None."""
    sink = _ListSink()

    async def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("modal connection lost")

    with _patched_provider(
        compile_side_effect=boom,
        invocation_sink=sink,
    ) as provider:
        provider.submit(_CANDIDATE_KERNEL).result(timeout=5.0)

    assert len(sink.records) == 1
    record = sink.records[0]
    assert isinstance(record, KernelBenchModalEvaluationRecord)
    assert record.reward is None


def test_sink_records_infrastructure_failure_when_evaluate_returns_none() -> None:
    """Lock-file race (evaluate returns None) → one record with reward=None."""
    sink = _ListSink()
    with _patched_provider(
        compile_return={"cache_dir": "/cache/x", "error": None},
        evaluate_return=None,
        invocation_sink=sink,
    ) as provider:
        provider.submit(_CANDIDATE_KERNEL).result(timeout=5.0)

    assert len(sink.records) == 1
    record = sink.records[0]
    assert isinstance(record, KernelBenchModalEvaluationRecord)
    assert record.reward is None


def test_no_sink_no_records() -> None:
    """Without a sink, the provider must complete the same code paths
    without raising — a sink is purely optional cost telemetry."""
    with _patched_provider(
        compile_return={"cache_dir": "/cache/x", "error": None},
        evaluate_return=_ok_exec_result(),
        invocation_sink=None,
    ) as provider:
        evaluation = provider.submit(_CANDIDATE_KERNEL).result(timeout=5.0)
    assert isinstance(evaluation.observation.feedback, SuccessFeedback)


def test_sink_one_record_per_submit_under_concurrency() -> None:
    """N concurrent submits → exactly N records. Asserts the sink path
    doesn't drop records under the asyncio loop's multiplexing."""
    sink = _ListSink()
    n_submits = 6
    with _patched_provider(
        compile_return={"cache_dir": "/cache/x", "error": None},
        evaluate_return=_ok_exec_result(),
        max_in_flight=3,
        invocation_sink=sink,
    ) as provider:
        futures = [provider.submit(_CANDIDATE_KERNEL) for _ in range(n_submits)]
        for f in futures:
            f.result(timeout=10.0)

    assert len(sink.records) == n_submits


