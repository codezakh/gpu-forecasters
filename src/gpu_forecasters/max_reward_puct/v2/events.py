"""Events that form the spine of a v2 max-reward PUCT search.

Every state transition is one of these events, appended to a durable log.
``SearchState`` is the pure fold of the log (see ``state.py``).

Shape decisions (iteration 2):

* **Per-candidate atomicity.** ``MutationRequested``/``MutationCompleted``
  name exactly one candidate, not a batch. This matches the atomic unit
  the v2-native providers expose (``submit(...) -> Future[str]``) and
  makes the durability claim honest: a logged Completed event means that
  one code was actually produced, and an un-terminated Requested event
  identifies *one* piece of work to re-dispatch on recovery.
* **Request events carry enough to re-dispatch.**
  ``MutationRequested.parent_ulid`` resolves to a ``Node`` in the archive
  (code + evaluation), and ``EvaluationRequested.code`` is embedded
  directly. Crash recovery needs no additional lookup outside the log.
* **Child ulids are pinned at ``EvaluationRequested``.** Not at
  completion. This keeps replay deterministic: the same log yields the
  same node identities.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Generic, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from ulid import ULID

from gpu_forecasters.hill_climbing.domain import Evaluation, Node, ObservationT


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class SearchInitialized(_Frozen, Generic[ObservationT]):
    kind: Literal["search_initialized"] = "search_initialized"
    root: Node[ObservationT]


class StepStarted(_Frozen):
    """Pins the parent selection for a step. The reducer trusts this;
    selection is not re-run on replay."""

    kind: Literal["step_started"] = "step_started"
    step: int
    parent_ulids: list[ULID]


class MutationRequested(_Frozen):
    """Declares intent to produce one mutation from one parent.

    Carries only ``parent_ulid``; the parent's code + evaluation are
    resolved via ``SearchState.archive`` at re-dispatch time.
    """

    kind: Literal["mutation_requested"] = "mutation_requested"
    request_id: str
    parent_ulid: ULID
    started_at: datetime | None = None


class MutationCompleted(_Frozen):
    kind: Literal["mutation_completed"] = "mutation_completed"
    request_id: str
    code: str
    completed_at: datetime | None = None


class MutationFailed(_Frozen):
    kind: Literal["mutation_failed"] = "mutation_failed"
    request_id: str
    reason: str
    completed_at: datetime | None = None


class EvaluationRequested(_Frozen):
    """Declares intent to evaluate one candidate under one parent.

    ``code`` is embedded so re-dispatch needs no other context.

    ``from_mutation_request_id`` ties this eval back to the mutation
    that produced ``code`` (when one exists). The reducer uses it to
    drain the mutation entry from ``in_flight_mutations`` — closing
    the crash-recovery window between ``MutationCompleted`` and the
    eval dispatch. ``None`` for evals not chained from a mutation
    (e.g., the bootstrap evaluation of a seed program).
    """

    kind: Literal["evaluation_requested"] = "evaluation_requested"
    request_id: str
    child_ulid: ULID
    parent_ulid: ULID
    code: str
    from_mutation_request_id: str | None = None
    started_at: datetime | None = None


class EvaluationCompleted(_Frozen, Generic[ObservationT]):
    kind: Literal["evaluation_completed"] = "evaluation_completed"
    request_id: str
    evaluation: Evaluation[ObservationT]
    completed_at: datetime | None = None


class EvaluationFailed(_Frozen):
    kind: Literal["evaluation_failed"] = "evaluation_failed"
    request_id: str
    reason: str
    completed_at: datetime | None = None


class StepCompleted(_Frozen):
    """Fence event. Triggers archive update + backprop in the reducer."""

    kind: Literal["step_completed"] = "step_completed"
    step: int


SearchEvent = Annotated[
    Union[
        SearchInitialized[ObservationT],
        StepStarted,
        MutationRequested,
        MutationCompleted,
        MutationFailed,
        EvaluationRequested,
        EvaluationCompleted[ObservationT],
        EvaluationFailed,
        StepCompleted,
    ],
    Field(discriminator="kind"),
]
"""Generic discriminated union — use for type annotations only. At
runtime (e.g. building a Pydantic ``TypeAdapter``), call
``search_event_adapter(ConcreteObservation)`` instead: Python's typing
machinery can't re-parameterize an already-parameterized
``Annotated[Union, ...]``."""


def search_event_adapter(
    observation_type: type[ObservationT],
) -> TypeAdapter[SearchEvent[ObservationT]]:
    """Build a ``TypeAdapter`` for the discriminated union of events,
    specialized on a concrete ``ObservationT``.

    Pydantic needs the concrete types to drive discriminator-based
    serialization and parsing. Use this anywhere you need to serialize
    or deserialize events to/from JSON. The returned adapter is typed
    so downstream reads are statically-checked end to end.
    """
    union = Annotated[
        Union[
            SearchInitialized[observation_type],  # type: ignore[valid-type]
            StepStarted,
            MutationRequested,
            MutationCompleted,
            MutationFailed,
            EvaluationRequested,
            EvaluationCompleted[observation_type],  # type: ignore[valid-type]
            EvaluationFailed,
            StepCompleted,
        ],
        Field(discriminator="kind"),
    ]
    return TypeAdapter(union)
