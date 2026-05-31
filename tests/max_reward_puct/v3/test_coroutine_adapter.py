"""Unit tests for ``CoroutineSpeedupEstimator``.

The adapter owns an asyncio event loop in a daemon thread between
``__enter__`` and ``__exit__``. We verify (1) ``submit`` returns a
working future, (2) multiple concurrent submits run in parallel, (3)
calling ``submit`` outside the context-manager scope raises, and (4)
``__exit__`` cancels in-flight tasks rather than waiting for them.
"""

from __future__ import annotations

import asyncio
import threading
import time

import pytest

from gpu_forecasters.landscape_map.v2 import (
    SUCCESS_BINS,
    HardwareContext,
    KernelImplementation,
    KernelRuntimeEstimate,
    KernelRuntimeQuery,
    KernelTaskInfo,
    LlmCallUsage,
    SpeedupBin,
)
from gpu_forecasters.max_reward_puct.v3.scoring_providers import (
    CoroutineSpeedupEstimator,
)


_HARDWARE = HardwareContext(
    device_name="test-cpu",
    compute_capability=(0, 0),
    total_global_memory_gb=0.0,
    multiprocessor_count=0,
    max_threads_per_multiprocessor=0,
    clock_rate_ghz=0.0,
    memory_clock_rate_ghz=0.0,
    memory_bus_width_bits=0,
)


def _query() -> KernelRuntimeQuery:
    return KernelRuntimeQuery(
        task=KernelTaskInfo(op_name="t", level_id=0, task_id=0),
        reference=KernelImplementation(
            kernel_name="r", code="ref", runtime_ms=None
        ),
        candidate=KernelImplementation(
            kernel_name="c", code="cand", runtime_ms=None
        ),
        hardware=_HARDWARE,
    )


def _uniform_estimate() -> KernelRuntimeEstimate:
    spread = 1.0 / len(SUCCESS_BINS)
    return KernelRuntimeEstimate(
        predicted_bin=SpeedupBin.MINOR_SLOWDOWN,
        bin_probabilities={b: spread for b in SUCCESS_BINS},
        reasoning="t",
        raw_probability_sum=1.0,
    )


class _SleepingEstimator:
    """Async estimator that sleeps for ``sleep_s`` then returns a fixed
    estimate. Tracks the threading.Event passed in, so tests can
    observe whether the coroutine actually ran."""

    def __init__(self, sleep_s: float, started: threading.Event) -> None:
        self._sleep_s = sleep_s
        self._started = started

    async def aestimate(
        self, query: KernelRuntimeQuery
    ) -> tuple[KernelRuntimeEstimate, LlmCallUsage | None]:
        del query
        self._started.set()
        await asyncio.sleep(self._sleep_s)
        return _uniform_estimate(), None


def test_submit_returns_estimate():
    started = threading.Event()
    inner = _SleepingEstimator(sleep_s=0.0, started=started)
    with CoroutineSpeedupEstimator(inner) as adapter:
        future = adapter.submit(_query())
        estimate, usage = future.result(timeout=5.0)
    assert estimate.predicted_bin == SpeedupBin.MINOR_SLOWDOWN
    assert usage is None
    assert started.is_set()


def test_concurrent_submits_run_in_parallel():
    """Two submits with sleeps should overlap, not serialize. We
    measure wall time and assert it's substantially below 2× sleep."""
    inner = _SleepingEstimator(sleep_s=0.5, started=threading.Event())
    with CoroutineSpeedupEstimator(inner) as adapter:
        t0 = time.monotonic()
        f1 = adapter.submit(_query())
        f2 = adapter.submit(_query())
        f1.result(timeout=5.0)
        f2.result(timeout=5.0)
        elapsed = time.monotonic() - t0
    # Serial would take ~1.0s; parallel should be ~0.5s + overhead.
    # Allow generous slack for CI variance.
    assert elapsed < 0.9, f"submits did not run concurrently (elapsed={elapsed:.3f}s)"


def test_submit_outside_context_raises():
    inner = _SleepingEstimator(sleep_s=0.0, started=threading.Event())
    adapter = CoroutineSpeedupEstimator(inner)
    with pytest.raises(RuntimeError, match="context manager"):
        adapter.submit(_query())


def test_exit_cancels_in_flight_tasks():
    """An in-flight forecast that's mid-await when __exit__ runs
    should be cancelled, not wait to natural completion."""
    started = threading.Event()
    inner = _SleepingEstimator(sleep_s=10.0, started=started)
    t0 = time.monotonic()
    with CoroutineSpeedupEstimator(inner, shutdown_timeout_s=2.0) as adapter:
        adapter.submit(_query())
        # Wait for the coroutine to actually start awaiting before we
        # exit; otherwise cancellation is racing the dispatch.
        assert started.wait(timeout=2.0)
    elapsed = time.monotonic() - t0
    # Should exit well before the 10s sleep completes.
    assert elapsed < 5.0, f"adapter waited for sleeping task (elapsed={elapsed:.3f}s)"
