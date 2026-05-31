"""Provider protocols for v3 search.

All three providers share the same shape: ``submit(...)`` returns a
``concurrent.futures.Future`` immediately, ``__enter__``/``__exit__``
manage lifecycle, and the driver waits on completed futures via
``concurrent.futures.wait(FIRST_COMPLETED)``. No asyncio appears in
the driver — providers that wrap async-native backends own their own
event loop internally (see
``gpu_forecasters.max_reward_puct.v3.scoring_providers.coroutine_adapter``
for the surrogate side).

Per-call atomic units: one ``submit`` yields one mutation, one
forecast, or one evaluation. No batch shapes at this seam — the
algorithm's atomic unit must agree with the log's atomic unit,
otherwise durability is a lie.
"""

from __future__ import annotations

from concurrent.futures import Future
from typing import Protocol, Self

from gpu_forecasters.hill_climbing.domain import (
    Evaluation,
    ObservationT,
)
from gpu_forecasters.landscape_map.v2 import (
    KernelRuntimeEstimate,
    KernelRuntimeQuery,
    LlmCallUsage,
)


class MutationProvider(Protocol[ObservationT]):
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


class EvaluationProvider(Protocol[ObservationT]):
    """Submits one evaluation at a time and returns a future.

    Same threading contract as ``MutationProvider``. Lifecycle methods
    own the Modal session (or equivalent).
    """

    def submit(self, program_code: str) -> Future[Evaluation[ObservationT]]: ...

    def __enter__(self) -> Self: ...

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None: ...


class SpeedupEstimator(Protocol):
    """Submits one surrogate forecast at a time and returns a future.

    Same shape as the other two providers. Async-native estimators (the
    v2 ``AsyncSpeedupEstimator`` family) are adapted to this protocol
    by ``CoroutineSpeedupEstimator``, which owns an asyncio loop in a
    daemon thread between ``__enter__`` and ``__exit__``.
    """

    def submit(
        self, query: KernelRuntimeQuery
    ) -> Future[tuple[KernelRuntimeEstimate, LlmCallUsage | None]]: ...

    def __enter__(self) -> Self: ...

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None: ...
