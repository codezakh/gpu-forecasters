"""Component interfaces for the theory-builder loop.

Two seams. Each is a Protocol; each has at least one v1 concrete and
can be swapped for testing or alternate implementations.

* ``ExperimentWorker.run(hypothesis) -> ExperimentResult`` — the inner
  search.
* ``WorldModelBuilder`` — proposes hypotheses and writes explanations.
  Both methods take domain objects directly: any rendering of the
  world model into a string is an internal concern of the concrete
  implementation, not part of the loop's contract.

No imports of LLM clients or Modal here. Concrete implementations live
in their own modules and import what they need.
"""

from __future__ import annotations

from typing import Protocol, Self

from gpu_forecasters.hill_climbing.domain import ObservationT
from gpu_forecasters.theory_builder.v1.domain import (
    Explanation,
    ExperimentResult,
    Hypothesis,
    WorldModel,
)


class ExperimentWorker(Protocol[ObservationT]):
    """Runs a small inner search seeded by a hypothesis.

    Used as a context manager: lifecycle methods own any backend
    session (Modal, LLM loop thread, ...). ``run`` is synchronous —
    the outer loop is single-threaded and blocking is fine.
    """

    def run(self, hypothesis: Hypothesis) -> ExperimentResult[ObservationT]: ...

    def __enter__(self) -> Self: ...

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None: ...


class WorldModelBuilder(Protocol[ObservationT]):
    """Proposes hypotheses and writes explanations.

    The two methods are deliberately stateless w.r.t. the world model:
    the loop hands the current ``WorldModel`` in each call. This keeps
    replay deterministic — there's no hidden builder state that can
    drift from the event log.

    ``propose_explanation`` returns the new world-model text alongside
    the ``Explanation``. The applier's output is logged so replay
    doesn't depend on the applier's implementation staying constant.
    """

    def propose_hypothesis(self, world_model: WorldModel) -> Hypothesis: ...

    def propose_explanation(
        self,
        world_model: WorldModel,
        hypothesis: Hypothesis,
        result: ExperimentResult[ObservationT],
    ) -> tuple[Explanation, str]: ...

    def __enter__(self) -> Self: ...

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None: ...


__all__ = [
    "ExperimentWorker",
    "WorldModelBuilder",
]
