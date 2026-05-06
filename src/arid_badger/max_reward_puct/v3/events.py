"""Sixteen event types that form the spine of a v3 max-reward PUCT search.

Every state transition the driver makes — provider call dispatched,
phase advanced, candidate settled, step completed — has a corresponding
event. The reducer (see ``state.py``) folds these events into
``SearchState``; nothing the algorithm cares about lives only in memory
or only in derived state.

Per-candidate events carry ``(step, request_id, parent_ulid)``;
per-parent barrier events carry ``(step, parent_ulid)``. Within those
keying schemes, ``(request_id, type)`` and ``(step, parent_ulid, type)``
are unique respectively. The append-only log's line position is the
universal temporal reference — there is no separate ``event_id``.

Reducer-derivable fields (archive size, parent rank, terminated request
ids, etc.) are deliberately *not* recorded here: that would create two
sources of truth and let them drift.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Generic, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter
from ulid import ULID

from arid_badger.hill_climbing.domain import Evaluation, Node, ObservationT
from arid_badger.landscape_map.v2 import (
    HardwareContext,
    KernelRuntimeEstimate,
    KernelTaskInfo,
    LlmCallUsage,
)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


# --- Per-search ----------------------------------------------------------


class SearchInitialized(_Frozen, Generic[ObservationT]):
    """Root candidate evaluated, archive initialized.

    Carries the per-search constants that condition every surrogate
    call (``kernel_task``, ``seed_reference_code``, ``hardware``). The
    spec requires the event log to be the system of record for
    everything the algorithm cares about; these three values directly
    determine forecasts, so they belong in the log alongside the root
    node. On resume, the driver validates its constructor args against
    these and refuses to continue with diverged context.
    """

    kind: Literal["search_initialized"] = "search_initialized"
    root: Node[ObservationT]
    kernel_task: KernelTaskInfo
    seed_reference_code: str
    hardware: HardwareContext


# --- Per-step ------------------------------------------------------------


class StepStarted(_Frozen):
    """PUCT chose this set of parents to expand for the step.

    ``selected_parent_scores`` is parallel to ``parent_ulids`` and
    records the composite PUCT score that drove each selection. Pinned
    here so analyses can recover the ranking the algorithm used.
    """

    kind: Literal["step_started"] = "step_started"
    step: int
    parent_ulids: list[ULID]
    selected_parent_scores: list[float]


class StepCompleted(_Frozen):
    """Fence event. Triggers archive update + backprop in the reducer."""

    kind: Literal["step_completed"] = "step_completed"
    step: int


# --- Per-candidate (lifecycle) ------------------------------------------


class MutationRequested(_Frozen):
    kind: Literal["mutation_requested"] = "mutation_requested"
    step: int
    request_id: ULID
    parent_ulid: ULID
    started_at: datetime | None = None


class MutationCompleted(_Frozen):
    kind: Literal["mutation_completed"] = "mutation_completed"
    step: int
    request_id: ULID
    parent_ulid: ULID
    code: str
    llm_usage: LlmCallUsage | None = None
    completed_at: datetime | None = None


class MutationFailed(_Frozen):
    kind: Literal["mutation_failed"] = "mutation_failed"
    step: int
    request_id: ULID
    parent_ulid: ULID
    reason: str
    completed_at: datetime | None = None


class ForecastRequested(_Frozen):
    kind: Literal["forecast_requested"] = "forecast_requested"
    step: int
    request_id: ULID
    parent_ulid: ULID
    code: str
    started_at: datetime | None = None


class ForecastCompleted(_Frozen):
    kind: Literal["forecast_completed"] = "forecast_completed"
    step: int
    request_id: ULID
    parent_ulid: ULID
    forecast: KernelRuntimeEstimate
    llm_usage: LlmCallUsage | None = None
    completed_at: datetime | None = None


class ForecastFailed(_Frozen):
    kind: Literal["forecast_failed"] = "forecast_failed"
    step: int
    request_id: ULID
    parent_ulid: ULID
    reason: str
    completed_at: datetime | None = None


class CandidateSelected(_Frozen):
    """Surrogate-filtered into the GPU-evaluation set.

    ``selection_score`` is the scalar produced by the configured
    ranking rule at the moment of selection. It is not redundant with
    the candidate's forecast: depending on the ranking rule the score
    can depend on archive state that drifts during the run (e.g.,
    empirical bin midpoints used to compute ``E[speedup]``). Recording
    it here pins the audit trail.
    """

    kind: Literal["candidate_selected"] = "candidate_selected"
    step: int
    request_id: ULID
    parent_ulid: ULID
    selection_score: float


class CandidateDeferred(_Frozen):
    """Surrogate-filtered out; settles with no GPU evaluation.

    ``selection_score`` carries the same scalar the ranking rule
    produced for this candidate so analyses can read the gap between
    the worst-selected and best-deferred candidate from the log
    directly.
    """

    kind: Literal["candidate_deferred"] = "candidate_deferred"
    step: int
    request_id: ULID
    parent_ulid: ULID
    selection_score: float


class EvaluationRequested(_Frozen):
    kind: Literal["evaluation_requested"] = "evaluation_requested"
    step: int
    request_id: ULID
    parent_ulid: ULID
    code: str
    started_at: datetime | None = None


class EvaluationCompleted(_Frozen, Generic[ObservationT]):
    kind: Literal["evaluation_completed"] = "evaluation_completed"
    step: int
    request_id: ULID
    parent_ulid: ULID
    evaluation: Evaluation[ObservationT]
    completed_at: datetime | None = None


class EvaluationFailed(_Frozen):
    kind: Literal["evaluation_failed"] = "evaluation_failed"
    step: int
    request_id: ULID
    parent_ulid: ULID
    reason: str
    completed_at: datetime | None = None


# --- Per-parent (phase barriers) ---------------------------------------


class MutationsDrained(_Frozen):
    """All of (step, parent)'s mutations have terminated."""

    kind: Literal["mutations_drained"] = "mutations_drained"
    step: int
    parent_ulid: ULID


class ForecastsDrained(_Frozen):
    """All of (step, parent)'s mutations and forecasts have terminated.
    Triggers selection in ``compute_pending_actions``."""

    kind: Literal["forecasts_drained"] = "forecasts_drained"
    step: int
    parent_ulid: ULID


class EvaluationsDrained(_Frozen):
    """All of (step, parent)'s selected candidates have terminal
    evaluation outcomes."""

    kind: Literal["evaluations_drained"] = "evaluations_drained"
    step: int
    parent_ulid: ULID


# --- Discriminated union -------------------------------------------------


SearchEvent = Annotated[
    Union[
        SearchInitialized[ObservationT],
        StepStarted,
        StepCompleted,
        MutationRequested,
        MutationCompleted,
        MutationFailed,
        ForecastRequested,
        ForecastCompleted,
        ForecastFailed,
        CandidateSelected,
        CandidateDeferred,
        EvaluationRequested,
        EvaluationCompleted[ObservationT],
        EvaluationFailed,
        MutationsDrained,
        ForecastsDrained,
        EvaluationsDrained,
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
    specialized on a concrete ``ObservationT``."""
    union = Annotated[
        Union[
            SearchInitialized[observation_type],  # type: ignore[valid-type]
            StepStarted,
            StepCompleted,
            MutationRequested,
            MutationCompleted,
            MutationFailed,
            ForecastRequested,
            ForecastCompleted,
            ForecastFailed,
            CandidateSelected,
            CandidateDeferred,
            EvaluationRequested,
            EvaluationCompleted[observation_type],  # type: ignore[valid-type]
            EvaluationFailed,
            MutationsDrained,
            ForecastsDrained,
            EvaluationsDrained,
        ],
        Field(discriminator="kind"),
    ]
    return TypeAdapter(union)
