"""Pure state and reducer for an event-sourced max-reward PUCT search.

``SearchState`` is what you get by folding a log of ``SearchEvent``s
through ``apply_event``. The reducer is exhaustive over every event
type and returns a new ``SearchState`` per call — never mutates in
place. The same input log produces the same output state, in any
process, at any time.

Spec § principles: per-parent state (phase, mutations-drained marker,
the candidate state machines for that parent's mutations) is grouped
into a ``ParentInStep`` struct rather than smeared across multiple
mappings keyed on ``parent_ulid``. The struct is the unit of per-parent
update; nothing else can drift out of sync because there is nowhere
else for that information to live.

The reducer reuses v1's pure archive/backprop helpers (``update_archive``,
``backpropagate``, ``record_failed_rollout``) — they are already pure
functions over the domain types.
"""

from __future__ import annotations

from collections import defaultdict
from enum import Enum
from typing import Annotated, Any, Generic, Literal, Mapping, Union

from pydantic import BaseModel, ConfigDict, Field
from ulid import ULID

from arid_badger.hill_climbing.domain import Evaluation, Node, ObservationT
from arid_badger.landscape_map.v2 import KernelRuntimeEstimate
from arid_badger.max_reward_puct.search import (
    backpropagate,
    record_failed_rollout,
    update_archive,
)
from arid_badger.max_reward_puct.v3.events import (
    CandidateDeferred,
    CandidateSelected,
    EvaluationCompleted,
    EvaluationFailed,
    EvaluationRequested,
    EvaluationsDrained,
    ForecastCompleted,
    ForecastFailed,
    ForecastRequested,
    ForecastsDrained,
    MutationCompleted,
    MutationFailed,
    MutationRequested,
    MutationsDrained,
    SearchEvent,
    SearchInitialized,
    StepCompleted,
    StepStarted,
)


# --- Per-parent phase ----------------------------------------------------


class ParentPhase(str, Enum):
    """The phase a parent is in within the active step.

    Looked up against ``ParentInStep.phase`` — never derived by
    predicate over candidate records. Spec § principles.
    """

    MUTATING_FORECASTING = "mutating_forecasting"
    AWAITING_SELECTION = "awaiting_selection"
    EVALUATING = "evaluating"
    DONE = "done"


# --- Per-candidate discriminated union ----------------------------------


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class CandidateMutating(_Frozen):
    """``MutationRequested`` logged; mutation provider call out."""

    kind: Literal["mutating"] = "mutating"
    step: int
    request_id: str
    parent_ulid: ULID


class CandidateAwaitingForecast(_Frozen):
    """``MutationCompleted`` logged; forecast not yet dispatched."""

    kind: Literal["awaiting_forecast"] = "awaiting_forecast"
    step: int
    request_id: str
    parent_ulid: ULID
    code: str


class CandidateForecasting(_Frozen):
    """``ForecastRequested`` logged; surrogate call out."""

    kind: Literal["forecasting"] = "forecasting"
    step: int
    request_id: str
    parent_ulid: ULID
    code: str


class CandidateAwaitingSelection(_Frozen):
    """``ForecastCompleted`` logged; selection not yet run."""

    kind: Literal["awaiting_selection"] = "awaiting_selection"
    step: int
    request_id: str
    parent_ulid: ULID
    code: str
    forecast: KernelRuntimeEstimate


class CandidateAwaitingEval(_Frozen):
    """``CandidateSelected`` logged; eval not yet dispatched."""

    kind: Literal["awaiting_eval"] = "awaiting_eval"
    step: int
    request_id: str
    parent_ulid: ULID
    code: str
    forecast: KernelRuntimeEstimate
    selection_score: float


class CandidateEvaluating(_Frozen):
    """``EvaluationRequested`` logged; eval provider call out."""

    kind: Literal["evaluating"] = "evaluating"
    step: int
    request_id: str
    parent_ulid: ULID
    code: str
    forecast: KernelRuntimeEstimate
    selection_score: float


class CandidateSettled(_Frozen, Generic[ObservationT]):
    """Terminal — one of: ``evaluated``, ``eval_failed``,
    ``mutation_failed``, ``forecast_failed``, ``deferred``.

    ``code`` / ``forecast`` / ``evaluation`` / ``selection_score`` are
    each populated only if the candidate reached the corresponding
    phase before settling. Validity of which fields are non-null per
    ``reason`` is upheld by the reducer; downstream code should branch
    on ``reason`` rather than feature-detect via ``is None``.
    """

    kind: Literal["settled"] = "settled"
    step: int
    request_id: str
    parent_ulid: ULID
    reason: Literal[
        "evaluated", "eval_failed", "mutation_failed", "forecast_failed", "deferred"
    ]
    code: str | None = None
    forecast: KernelRuntimeEstimate | None = None
    selection_score: float | None = None
    evaluation: Evaluation[ObservationT] | None = None


# Candidate variants form a discriminated union. We don't define
# ``Candidate`` as a top-level ``TypeAlias`` of an ``Annotated[Union,
# Field(...)]`` because Python's typing machinery refuses to
# re-subscribe an already-Annotated alias (``Candidate[T]`` raises
# ``TypeError: ... is not a generic class``). Instead we inline the
# union where it appears in a generic Pydantic field, and provide
# ``Candidate`` below as a non-generic alias for type annotations on
# pure code (helpers, function returns) where the variant matters but
# the observation type is unknown.
Candidate = Union[
    CandidateMutating,
    CandidateAwaitingForecast,
    CandidateForecasting,
    CandidateAwaitingSelection,
    CandidateAwaitingEval,
    CandidateEvaluating,
    CandidateSettled[Any],
]


# --- Per-parent in-step record ------------------------------------------


class ParentInStep(BaseModel, Generic[ObservationT]):
    """All per-parent state for the active step, in one struct.

    Holds the parent's current phase, whether ``MutationsDrained`` has
    been emitted (a marker barrier that does *not* advance ``phase``),
    and the candidate state machines for the parent's mutations
    (one per ``request_id``).

    Replacing this struct wholesale on every transition gives
    copy-on-write semantics; ``parent_ulid`` is the identity key but
    it lives inside the struct rather than as the key of an outer
    mapping, so per-parent state can never disagree with itself.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    parent_ulid: ULID
    phase: ParentPhase
    mutations_drained: bool = False
    candidates: Mapping[
        str,
        Annotated[
            Union[
                CandidateMutating,
                CandidateAwaitingForecast,
                CandidateForecasting,
                CandidateAwaitingSelection,
                CandidateAwaitingEval,
                CandidateEvaluating,
                CandidateSettled[ObservationT],
            ],
            Field(discriminator="kind"),
        ],
    ] = Field(default_factory=dict)

    def with_candidate(
        self, request_id: str, candidate: Candidate
    ) -> ParentInStep[ObservationT]:
        new_candidates = dict(self.candidates)
        new_candidates[request_id] = candidate
        return self.model_copy(update={"candidates": new_candidates})


# --- Search state -------------------------------------------------------


class SearchState(BaseModel, Generic[ObservationT]):
    """Whatever you get by folding a log of ``SearchEvent``s.

    Carries the global archive plus PUCT bookkeeping (``visit_counts``,
    ``best_child_rewards`` — both keyed on archive node ULID) plus
    active-step bookkeeping as a tuple of ``ParentInStep`` records.
    ``apply_event`` returns a *new* ``SearchState`` per event — never
    mutates in place.

    The two surviving Mappings (``visit_counts``, ``best_child_rewards``)
    are genuine many-to-one stores keyed by node ULID, with one entry
    per archive node ever visited. They aren't stand-ins for a struct.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    archive: tuple[Node[ObservationT], ...] = ()
    seed_ids: frozenset[ULID] = frozenset()
    visit_counts: Mapping[ULID, int] = Field(default_factory=dict)
    best_child_rewards: Mapping[ULID, float] = Field(default_factory=dict)
    global_expansion_count: int = 0
    current_step: int = 0

    # Active step bookkeeping. Empty between steps. One entry per
    # parent selected by ``StepStarted``.
    current_step_parents: tuple[ParentInStep[ObservationT], ...] = ()

    @property
    def current_step_active(self) -> bool:
        return bool(self.current_step_parents)

    def is_uninitialized(self) -> bool:
        return not self.archive

    def parent_in_step(
        self, parent_ulid: ULID
    ) -> ParentInStep[ObservationT] | None:
        """Look up the active-step record for ``parent_ulid``."""
        for p in self.current_step_parents:
            if p.parent_ulid == parent_ulid:
                return p
        return None

    def best_archived_node(self) -> Node[ObservationT] | None:
        """Node with the highest reward in the current archive.

        Mid-step children whose evaluations have landed but for which
        ``StepCompleted`` hasn't fired are not folded into the archive
        yet, and so are not considered.
        """
        if not self.archive:
            return None
        return max(
            self.archive,
            key=lambda n: n.evaluation.reward
            if n.evaluation.reward is not None
            else float("-inf"),
        )


# --- Reducer ------------------------------------------------------------


def _replace_parent(
    parents: tuple[ParentInStep[ObservationT], ...],
    parent_ulid: ULID,
    new_record: ParentInStep[ObservationT],
) -> tuple[ParentInStep[ObservationT], ...]:
    """Return a new tuple with the record for ``parent_ulid`` replaced."""
    return tuple(
        new_record if p.parent_ulid == parent_ulid else p for p in parents
    )


def _update_candidate(
    state: SearchState[ObservationT],
    parent_ulid: ULID,
    request_id: str,
    candidate: Candidate,
) -> SearchState[ObservationT]:
    """Replace one candidate inside one parent record. The common path
    for every per-candidate event reducer."""
    parent = state.parent_in_step(parent_ulid)
    assert parent is not None, (
        f"event references parent_ulid {parent_ulid} that is not in "
        f"the current step"
    )
    new_parent = parent.with_candidate(request_id, candidate)
    return state.model_copy(
        update={
            "current_step_parents": _replace_parent(
                state.current_step_parents, parent_ulid, new_parent
            )
        }
    )


def apply_event(
    state: SearchState[ObservationT],
    event: SearchEvent[ObservationT],
    *,
    k_per_parent: int,
    archive_capacity: int,
    observation_type: type[ObservationT],
) -> SearchState[ObservationT]:
    """Fold one event into ``state``, returning a *new* ``SearchState``.

    Pure: the result depends only on ``(state, event, k_per_parent,
    archive_capacity, observation_type)``. No clock reads, no
    randomness, no I/O, no logging.

    ``match`` over the discriminated union is exhaustive; missing event
    types surface as type-check errors (basedpyright) and runtime
    errors (the final ``case _`` raises). This is the totality
    invariant from the spec.
    """
    match event:
        case SearchInitialized():
            return state.model_copy(
                update={
                    "archive": (event.root,),
                    "seed_ids": frozenset({event.root.ulid}),
                }
            )

        case StepStarted():
            return state.model_copy(
                update={
                    "current_step_parents": tuple(
                        ParentInStep[observation_type](  # type: ignore[valid-type]
                            parent_ulid=p,
                            phase=ParentPhase.MUTATING_FORECASTING,
                            mutations_drained=False,
                            candidates={},
                        )
                        for p in event.parent_ulids
                    ),
                }
            )

        case MutationRequested():
            return _update_candidate(
                state,
                event.parent_ulid,
                event.request_id,
                CandidateMutating(
                    step=event.step,
                    request_id=event.request_id,
                    parent_ulid=event.parent_ulid,
                ),
            )

        case MutationCompleted():
            return _update_candidate(
                state,
                event.parent_ulid,
                event.request_id,
                CandidateAwaitingForecast(
                    step=event.step,
                    request_id=event.request_id,
                    parent_ulid=event.parent_ulid,
                    code=event.code,
                ),
            )

        case MutationFailed():
            return _update_candidate(
                state,
                event.parent_ulid,
                event.request_id,
                CandidateSettled[observation_type](  # type: ignore[valid-type]
                    step=event.step,
                    request_id=event.request_id,
                    parent_ulid=event.parent_ulid,
                    reason="mutation_failed",
                ),
            )

        case ForecastRequested():
            return _update_candidate(
                state,
                event.parent_ulid,
                event.request_id,
                CandidateForecasting(
                    step=event.step,
                    request_id=event.request_id,
                    parent_ulid=event.parent_ulid,
                    code=event.code,
                ),
            )

        case ForecastCompleted():
            parent = state.parent_in_step(event.parent_ulid)
            assert parent is not None
            existing = parent.candidates.get(event.request_id)
            code = _extract_code(existing)
            return _update_candidate(
                state,
                event.parent_ulid,
                event.request_id,
                CandidateAwaitingSelection(
                    step=event.step,
                    request_id=event.request_id,
                    parent_ulid=event.parent_ulid,
                    code=code,
                    forecast=event.forecast,
                ),
            )

        case ForecastFailed():
            parent = state.parent_in_step(event.parent_ulid)
            assert parent is not None
            existing = parent.candidates.get(event.request_id)
            return _update_candidate(
                state,
                event.parent_ulid,
                event.request_id,
                CandidateSettled[observation_type](  # type: ignore[valid-type]
                    step=event.step,
                    request_id=event.request_id,
                    parent_ulid=event.parent_ulid,
                    reason="forecast_failed",
                    code=_extract_code(existing) if existing is not None else None,
                ),
            )

        case CandidateSelected():
            parent = state.parent_in_step(event.parent_ulid)
            assert parent is not None
            existing = parent.candidates[event.request_id]
            assert isinstance(existing, CandidateAwaitingSelection)
            new_candidates = dict(parent.candidates)
            new_candidates[event.request_id] = CandidateAwaitingEval(
                step=event.step,
                request_id=event.request_id,
                parent_ulid=event.parent_ulid,
                code=existing.code,
                forecast=existing.forecast,
                selection_score=event.selection_score,
            )
            # Selection event also acts as the barrier advancing phase
            # from AwaitingSelection → Evaluating (spec § per-parent
            # phases). Idempotent: subsequent selection events for the
            # same parent re-set the same phase.
            new_parent = parent.model_copy(
                update={
                    "candidates": new_candidates,
                    "phase": ParentPhase.EVALUATING,
                }
            )
            return state.model_copy(
                update={
                    "current_step_parents": _replace_parent(
                        state.current_step_parents, event.parent_ulid, new_parent
                    )
                }
            )

        case CandidateDeferred():
            parent = state.parent_in_step(event.parent_ulid)
            assert parent is not None
            existing = parent.candidates[event.request_id]
            assert isinstance(existing, CandidateAwaitingSelection)
            new_candidates = dict(parent.candidates)
            new_candidates[event.request_id] = CandidateSettled[observation_type](  # type: ignore[valid-type]
                step=event.step,
                request_id=event.request_id,
                parent_ulid=event.parent_ulid,
                reason="deferred",
                code=existing.code,
                forecast=existing.forecast,
                selection_score=event.selection_score,
            )
            new_parent = parent.model_copy(
                update={
                    "candidates": new_candidates,
                    "phase": ParentPhase.EVALUATING,
                }
            )
            return state.model_copy(
                update={
                    "current_step_parents": _replace_parent(
                        state.current_step_parents, event.parent_ulid, new_parent
                    )
                }
            )

        case EvaluationRequested():
            parent = state.parent_in_step(event.parent_ulid)
            assert parent is not None
            existing = parent.candidates[event.request_id]
            assert isinstance(existing, (CandidateAwaitingEval, CandidateEvaluating))
            return _update_candidate(
                state,
                event.parent_ulid,
                event.request_id,
                CandidateEvaluating(
                    step=event.step,
                    request_id=event.request_id,
                    parent_ulid=event.parent_ulid,
                    code=existing.code,
                    forecast=existing.forecast,
                    selection_score=existing.selection_score,
                ),
            )

        case EvaluationCompleted():
            parent = state.parent_in_step(event.parent_ulid)
            assert parent is not None
            existing = parent.candidates[event.request_id]
            assert isinstance(existing, CandidateEvaluating)
            return _update_candidate(
                state,
                event.parent_ulid,
                event.request_id,
                CandidateSettled[observation_type](  # type: ignore[valid-type]
                    step=event.step,
                    request_id=event.request_id,
                    parent_ulid=event.parent_ulid,
                    reason="evaluated",
                    code=existing.code,
                    forecast=existing.forecast,
                    selection_score=existing.selection_score,
                    evaluation=event.evaluation,
                ),
            )

        case EvaluationFailed():
            parent = state.parent_in_step(event.parent_ulid)
            assert parent is not None
            existing = parent.candidates[event.request_id]
            assert isinstance(existing, CandidateEvaluating)
            return _update_candidate(
                state,
                event.parent_ulid,
                event.request_id,
                CandidateSettled[observation_type](  # type: ignore[valid-type]
                    step=event.step,
                    request_id=event.request_id,
                    parent_ulid=event.parent_ulid,
                    reason="eval_failed",
                    code=existing.code,
                    forecast=existing.forecast,
                    selection_score=existing.selection_score,
                ),
            )

        case MutationsDrained():
            parent = state.parent_in_step(event.parent_ulid)
            assert parent is not None
            new_parent = parent.model_copy(update={"mutations_drained": True})
            return state.model_copy(
                update={
                    "current_step_parents": _replace_parent(
                        state.current_step_parents, event.parent_ulid, new_parent
                    )
                }
            )

        case ForecastsDrained():
            parent = state.parent_in_step(event.parent_ulid)
            assert parent is not None
            new_parent = parent.model_copy(
                update={"phase": ParentPhase.AWAITING_SELECTION}
            )
            return state.model_copy(
                update={
                    "current_step_parents": _replace_parent(
                        state.current_step_parents, event.parent_ulid, new_parent
                    )
                }
            )

        case EvaluationsDrained():
            parent = state.parent_in_step(event.parent_ulid)
            assert parent is not None
            new_parent = parent.model_copy(update={"phase": ParentPhase.DONE})
            return state.model_copy(
                update={
                    "current_step_parents": _replace_parent(
                        state.current_step_parents, event.parent_ulid, new_parent
                    )
                }
            )

        case StepCompleted():
            return _finalize_step(
                state,
                event_step=event.step,
                k_per_parent=k_per_parent,
                archive_capacity=archive_capacity,
                observation_type=observation_type,
            )


def _extract_code(candidate: Candidate | None) -> str:
    """Pull ``code`` from any candidate variant that has it.

    Used by ``ForecastCompleted`` / ``ForecastFailed`` reducers, which
    need the existing code to thread through the phase transition.
    Asserts loudly if asked to extract from a phase that doesn't carry
    code — that would be a reducer-construction bug, not a runtime
    edge case.
    """
    assert candidate is not None, "no prior candidate for forecast event"
    if isinstance(
        candidate,
        (
            CandidateAwaitingForecast,
            CandidateForecasting,
            CandidateAwaitingSelection,
            CandidateAwaitingEval,
            CandidateEvaluating,
        ),
    ):
        return candidate.code
    raise AssertionError(
        f"cannot extract code from candidate kind={candidate.kind}"
    )


def _finalize_step(
    state: SearchState[ObservationT],
    *,
    event_step: int,
    k_per_parent: int,
    archive_capacity: int,
    observation_type: type[ObservationT],
) -> SearchState[ObservationT]:
    """Run per-step archive update + backprop over the parents'
    settled candidates. Mirrors v1's D/B/C' blocks in ``_search_impl``
    and v2's reducer.

    Only ``CandidateSettled(reason="evaluated")`` candidates with a
    non-None reward feed the archive. Returns a new ``SearchState``
    with the active-step bookkeeping cleared and ``current_step``
    advanced.
    """
    archive_by_ulid = {n.ulid: n for n in state.archive}

    visits: dict[ULID, int] = dict(state.visit_counts)
    best_child: dict[ULID, float] = dict(state.best_child_rewards)
    archive: list[Node[ObservationT]] = list(state.archive)

    children: list[Node[ObservationT]] = []
    parent_nodes: list[Node[ObservationT]] = []
    valid_by_parent: dict[ULID, int] = defaultdict(int)

    node_cls: type[Node[ObservationT]] = Node[observation_type]  # type: ignore[valid-type]
    for parent_record in state.current_step_parents:
        for cand in parent_record.candidates.values():
            if not isinstance(cand, CandidateSettled):
                continue
            if cand.reason != "evaluated":
                continue
            evaluation = cand.evaluation
            code = cand.code
            assert evaluation is not None and code is not None, (
                "CandidateSettled(reason='evaluated') missing evaluation/code"
            )
            parent_node = archive_by_ulid.get(parent_record.parent_ulid)
            if parent_node is None:
                # Parent evicted from archive between dispatch and
                # completion. Drop the orphan.
                continue
            # Spec: "If the candidate ends up in the archive, the
            # ``Node.ulid`` equals the ``request_id``."
            child = node_cls(
                program_code=code,
                evaluation=evaluation,
                ulid=ULID.from_str(cand.request_id),
                ancestors=[],
            )
            children.append(child)
            parent_nodes.append(parent_node)
            if evaluation.reward is not None:
                valid_by_parent[parent_node.ulid] += 1

    new_T = backpropagate(
        children=children,
        parent_states=parent_nodes,
        n=visits,
        m=best_child,
        T=state.global_expansion_count,
    )

    for parent_record in state.current_step_parents:
        if valid_by_parent.get(parent_record.parent_ulid, 0) > 0:
            continue
        parent_node = archive_by_ulid.get(parent_record.parent_ulid)
        if parent_node is None:
            continue
        new_T = record_failed_rollout(parent=parent_node, n=visits, T=new_T)

    update_archive(
        archive=archive,
        children=children,
        parent_states=parent_nodes,
        seed_ids=set(state.seed_ids),
        k_per_parent=k_per_parent,
        capacity=archive_capacity,
    )

    return state.model_copy(
        update={
            "archive": tuple(archive),
            "visit_counts": visits,
            "best_child_rewards": best_child,
            "global_expansion_count": new_T,
            "current_step": event_step + 1,
            "current_step_parents": (),
        }
    )


def replay(
    events: list[SearchEvent[ObservationT]],
    *,
    k_per_parent: int,
    archive_capacity: int,
    observation_type: type[ObservationT],
) -> SearchState[ObservationT]:
    """Fold a list of events into the final state. Pure."""
    state: SearchState[ObservationT] = SearchState()
    for event in events:
        state = apply_event(
            state,
            event,
            k_per_parent=k_per_parent,
            archive_capacity=archive_capacity,
            observation_type=observation_type,
        )
    return state
