"""Async provider protocols for v3 search.

Mutation and evaluation providers carry over unchanged from v2: one
``submit(...)`` per atomic unit, returning a ``concurrent.futures.Future``.
The surrogate side (``AsyncSpeedupEstimator``) is async-await; the
driver bridges via a background asyncio loop and
``run_coroutine_threadsafe``.
"""

from __future__ import annotations

from concurrent.futures import Future
from typing import Protocol, Self

from arid_badger.hill_climbing.domain import (
    Evaluation,
    ObservationT,
)


class AsyncMutationProvider(Protocol[ObservationT]):
    """Submits one mutation at a time and returns a future.

    Implementations must be safe to call ``submit`` concurrently from
    multiple threads (the driver does this). Lifecycle setup/teardown
    runs in ``__enter__``/``__exit__`` — e.g. starting/stopping an
    internal asyncio loop.
    """

    def submit(
        self,
        parent_code: str,
        evaluation: Evaluation[ObservationT],
    ) -> Future[str]: ...

    def __enter__(self) -> Self: ...

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None: ...


class AsyncEvaluationProvider(Protocol[ObservationT]):
    """Submits one evaluation at a time and returns a future.

    Same threading contract as ``AsyncMutationProvider``. Lifecycle
    methods own the Modal session (or equivalent).
    """

    def submit(self, program_code: str) -> Future[Evaluation[ObservationT]]: ...

    def __enter__(self) -> Self: ...

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None: ...
