"""Pure state and reducer for an event-sourced max-reward PUCT search.

``SearchState`` is whatever you get by folding a log of ``SearchEvent``s
through ``apply_event``. No I/O, no LLM calls, no Modal — just data.

The reducer reuses v1's pure archive/backprop helpers
(``update_archive``, ``backpropagate``, ``record_failed_rollout``). They
are already pure functions over the domain types; duplicating them
would create two places that can drift.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Generic

from pydantic import BaseModel, ConfigDict, Field
from ulid import ULID

from arid_badger.hill_climbing.domain import Evaluation, Node, ObservationT
from arid_badger.max_reward_puct.search import (
    backpropagate,
    record_failed_rollout,
    update_archive,
)
from arid_badger.max_reward_puct.v2.events import (
    EvaluationCompleted,
    EvaluationFailed,
    EvaluationRequested,
    MutationCompleted,
    MutationFailed,
    MutationRequested,
    SearchEvent,
    SearchInitialized,
    StepCompleted,
    StepStarted,
)


class MutationInFlight(BaseModel):
    """A mutation that has been requested but whose downstream
    ``EvaluationRequested`` has not yet been logged.

    ``code`` is set on ``MutationCompleted`` and remains until the
    paired ``EvaluationRequested`` drains the entry. This closes the
    durability gap between mutation success and eval dispatch:
    crash recovery sees ``code is not None`` and dispatches the eval;
    crash recovery sees ``code is None`` and re-runs the mutation.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    parent_ulid: ULID
    code: str | None = None


class PendingChild(BaseModel, Generic[ObservationT]):
    """A child whose evaluation has been requested but not yet folded
    into the archive. Populated by ``EvaluationRequested``; updated by
    ``EvaluationCompleted`` (fills ``evaluation``) or
    ``EvaluationFailed`` (sets ``failed=True``). Drained at
    ``StepCompleted``.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    child_ulid: ULID
    parent_ulid: ULID
    code: str
    evaluation: Evaluation[ObservationT] | None = None
    failed: bool = False


class SearchState(BaseModel, Generic[ObservationT]):
    """Everything needed to continue a search. A pure fold over the
    event log produces this."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    archive: list[Node[ObservationT]] = Field(default_factory=list)
    seed_ids: set[ULID] = Field(default_factory=set)
    visit_counts: dict[ULID, int] = Field(default_factory=dict)
    best_child_rewards: dict[ULID, float] = Field(default_factory=dict)
    global_expansion_count: int = 0

    # Number of steps *completed*. 0 means no ``StepCompleted`` has
    # been seen yet. The search's next step is ``current_step``.
    current_step: int = 0

    # --- Within-step bookkeeping. All cleared on ``StepCompleted``. ---

    # The parents pinned by ``StepStarted`` for this step. Empty
    # outside a step. Non-empty on startup => we crashed mid-step and
    # need to recover.
    step_parent_ulids: list[ULID] = Field(default_factory=list)

    # Mutation requests that have been issued but whose downstream
    # ``EvaluationRequested`` has not yet been logged. Lifecycle:
    #   * ``MutationRequested``  → entry inserted, ``code=None``.
    #   * ``MutationCompleted``  → ``code`` filled in.
    #   * ``MutationFailed``     → entry popped (mutation aborted).
    #   * ``EvaluationRequested`` (linked) → entry popped (chain closed).
    # On recovery: entries with ``code is None`` need a fresh mutation
    # submit; entries with ``code is not None`` need an eval dispatch.
    in_flight_mutations: dict[str, MutationInFlight] = Field(default_factory=dict)

    # Evaluation requests. Each entry is a ``PendingChild`` that
    # accumulates status as its terminal event lands. Completed and
    # failed entries stay in the map until ``StepCompleted`` because
    # they still feed the archive fold. Re-dispatch on recovery keys
    # off the ``pending`` status (neither ``evaluation`` nor ``failed``
    # set).
    in_flight_children: dict[str, PendingChild[ObservationT]] = Field(
        default_factory=dict
    )

    def best_archived_node(self) -> Node[ObservationT] | None:
        """Node with the highest reward in the current archive.

        Note: this only inspects ``archive``. Mid-step, children whose
        evaluations have landed but for which ``StepCompleted`` hasn't
        fired yet live in ``in_flight_children`` and are NOT considered
        — they haven't been folded in yet.
        """
        if not self.archive:
            return None
        return max(
            self.archive,
            key=lambda n: n.evaluation.reward
            if n.evaluation.reward is not None
            else float("-inf"),
        )


def apply_event(
    state: SearchState[ObservationT],
    event: SearchEvent[ObservationT],
    *,
    k_per_parent: int,
    archive_capacity: int,
    observation_type: type[ObservationT],
) -> SearchState[ObservationT]:
    """Fold one event into ``state``. Pure (state is mutated in place
    as a perf concession; the result depends only on the inputs).

    ``k_per_parent`` and ``archive_capacity`` parameterize the
    ``StepCompleted`` fold. Passing them per-call keeps the reducer
    stateless: the same function can fold any log given the same
    config.
    """
    match event:
        case SearchInitialized():
            state.archive = [event.root]
            state.seed_ids = {event.root.ulid}

        case StepStarted():
            state.step_parent_ulids = list(event.parent_ulids)
            state.in_flight_mutations = {}
            state.in_flight_children = {}

        case MutationRequested():
            state.in_flight_mutations[event.request_id] = MutationInFlight(
                parent_ulid=event.parent_ulid
            )

        case MutationCompleted():
            entry = state.in_flight_mutations.get(event.request_id)
            if entry is not None:
                entry.code = event.code

        case MutationFailed():
            state.in_flight_mutations.pop(event.request_id, None)

        case EvaluationRequested():
            if event.from_mutation_request_id is not None:
                state.in_flight_mutations.pop(
                    event.from_mutation_request_id, None
                )
            state.in_flight_children[event.request_id] = PendingChild[
                ObservationT
            ](
                child_ulid=event.child_ulid,
                parent_ulid=event.parent_ulid,
                code=event.code,
            )

        case EvaluationCompleted():
            pending = state.in_flight_children.get(event.request_id)
            if pending is not None:
                pending.evaluation = event.evaluation

        case EvaluationFailed():
            pending = state.in_flight_children.get(event.request_id)
            if pending is not None:
                pending.failed = True

        case StepCompleted():
            _finalize_step(
                state,
                k_per_parent=k_per_parent,
                archive_capacity=archive_capacity,
                observation_type=observation_type,
            )
            state.current_step = event.step + 1

    return state


def _finalize_step(
    state: SearchState[ObservationT],
    *,
    k_per_parent: int,
    archive_capacity: int,
    observation_type: type[ObservationT],
) -> None:
    """Run per-step archive update + backprop over the in-flight
    buffer. Mirrors the v1 D/B/C' blocks in ``_search_impl``.
    """
    archive_by_ulid = {n.ulid: n for n in state.archive}

    children: list[Node[ObservationT]] = []
    parent_nodes: list[Node[ObservationT]] = []
    valid_by_parent: dict[ULID, int] = defaultdict(int)

    for pending in state.in_flight_children.values():
        # Untermintated pending children are dropped at step fence.
        # In a clean run this never happens; under recovery the
        # driver re-dispatches pending entries *before* the fence, so
        # it still won't happen. Defensive skip.
        if pending.failed or pending.evaluation is None:
            continue
        parent_node = archive_by_ulid.get(pending.parent_ulid)
        if parent_node is None:
            # Parent got evicted from the archive between dispatch and
            # completion. Drop the orphan.
            continue
        # See note in driver: must subscribe with the runtime concrete
        # class or Pydantic defaults the TypeVar and drops fields.
        node_cls: type[Node[ObservationT]] = Node[observation_type]  # type: ignore[valid-type]
        child: Node[ObservationT] = node_cls(
            program_code=pending.code,
            evaluation=pending.evaluation,
            ulid=pending.child_ulid,
            ancestors=[],
        )
        children.append(child)
        parent_nodes.append(parent_node)
        if pending.evaluation.reward is not None:
            valid_by_parent[parent_node.ulid] += 1

    state.global_expansion_count = backpropagate(
        children=children,
        parent_states=parent_nodes,
        n=state.visit_counts,
        m=state.best_child_rewards,
        T=state.global_expansion_count,
    )

    # Any step-parent that produced zero valid children gets the
    # failed-rollout decay. Covers mutation failure, evaluation
    # failure, and all-None evaluations uniformly.
    for parent_ulid in state.step_parent_ulids:
        if valid_by_parent.get(parent_ulid, 0) > 0:
            continue
        parent_node = archive_by_ulid.get(parent_ulid)
        if parent_node is None:
            continue
        state.global_expansion_count = record_failed_rollout(
            parent=parent_node,
            n=state.visit_counts,
            T=state.global_expansion_count,
        )

    update_archive(
        archive=state.archive,
        children=children,
        parent_states=parent_nodes,
        seed_ids=state.seed_ids,
        k_per_parent=k_per_parent,
        capacity=archive_capacity,
    )

    state.step_parent_ulids = []
    state.in_flight_mutations = {}
    state.in_flight_children = {}


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
        _ = apply_event(
            state,
            event,
            k_per_parent=k_per_parent,
            archive_capacity=archive_capacity,
            observation_type=observation_type,
        )
    return state
