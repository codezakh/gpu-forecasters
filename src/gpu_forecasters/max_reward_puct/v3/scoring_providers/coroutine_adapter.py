"""Adapter from v2 ``AsyncSpeedupEstimator`` (coroutine-based) to v3
``SpeedupEstimator`` (futures-based).

The v2 surrogate family (litellm, tinker, abstain, ...) exposes
``async def aestimate(query) -> tuple[KernelRuntimeEstimate,
LlmCallUsage | None]``. The v3 driver consumes a futures-based
``SpeedupEstimator``, mirroring the mutation and evaluation provider
shape.

This adapter owns one asyncio event loop in a daemon thread for the
duration of its context-manager scope. ``submit`` schedules the
inner coroutine on that loop via ``asyncio.run_coroutine_threadsafe``
and returns the resulting ``concurrent.futures.Future``. On
``__exit__`` it cancels any in-flight tasks before stopping the loop.

The adapter is the only place in v3 that imports ``asyncio`` —
``SearchDriver`` is asyncio-free.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import Future
from typing import Self

from gpu_forecasters.landscape_map.v2 import (
    AsyncSpeedupEstimator,
    KernelRuntimeEstimate,
    KernelRuntimeQuery,
    LlmCallUsage,
)


class CoroutineSpeedupEstimator:
    """Wraps an ``AsyncSpeedupEstimator`` to satisfy v3
    ``SpeedupEstimator``.

    Single-instance, single-loop, single-thread. Not safe to share an
    instance across multiple driver runs in different threads (each
    driver run should construct its own).
    """

    def __init__(
        self, inner: AsyncSpeedupEstimator, *, shutdown_timeout_s: float = 10.0
    ) -> None:
        self._inner = inner
        self._shutdown_timeout_s = shutdown_timeout_s
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None

    def __enter__(self) -> Self:
        loop = asyncio.new_event_loop()
        thread = threading.Thread(
            target=loop.run_forever,
            name="v3-surrogate-loop",
            daemon=True,
        )
        thread.start()
        self._loop = loop
        self._thread = thread
        return self

    def __exit__(
        self, exc_type: object, exc_val: object, exc_tb: object
    ) -> None:
        loop = self._loop
        thread = self._thread
        if loop is not None:
            # Cancel any tasks the inner estimator scheduled and is
            # still awaiting (e.g. an httpx request mid-flight). Each
            # task gets a chance to honor the cancellation before the
            # loop stops; ones that don't honor it are abandoned with
            # the loop.
            cancel_event = threading.Event()

            def _cancel_pending() -> None:
                for task in asyncio.all_tasks(loop):
                    task.cancel()
                cancel_event.set()

            loop.call_soon_threadsafe(_cancel_pending)
            cancel_event.wait(timeout=self._shutdown_timeout_s)
            loop.call_soon_threadsafe(loop.stop)
        if thread is not None:
            thread.join(timeout=self._shutdown_timeout_s)
        self._loop = None
        self._thread = None

    def submit(
        self, query: KernelRuntimeQuery
    ) -> Future[tuple[KernelRuntimeEstimate, LlmCallUsage | None]]:
        if self._loop is None:
            raise RuntimeError(
                "CoroutineSpeedupEstimator must be entered as a context "
                "manager before submit() can be called"
            )
        return asyncio.run_coroutine_threadsafe(
            self._inner.aestimate(query), self._loop
        )
