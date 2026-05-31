"""Async provider protocols for v2 search.

The search's view of infrastructure is minimal:

* ``submit(...)`` returns a ``concurrent.futures.Future`` immediately.
* ``Future.result()`` blocks until the work is done.
* ``__enter__``/``__exit__`` manage any stateful backend lifecycle
  (Modal session, asyncio loop thread, ...).

Per-candidate atomic units: ``submit(parent_code, evaluation)`` for
mutations yields *one* code; ``submit(code)`` for evaluations yields
*one* evaluation. No batch shapes at this seam. That is the whole
point of v2 — the inner call's atomic unit must agree with the
log's atomic unit, otherwise durability is a lie.

The search itself does not own an ``Executor``. Providers own whatever
concurrency they need internally.
"""

from __future__ import annotations

from typing import Protocol, Self

from concurrent.futures import Future

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
