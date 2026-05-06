"""Pure decision function for v3 max-reward PUCT.

``compute_pending_actions(state, config)`` is the only place the
algorithm makes decisions about what to dispatch next. Pure: it reads
``state``, returns an ``Actions`` value, and does no I/O, no clock
reads, no randomness. The driver loop only executes those actions and
waits for completions.

Spec § principles: normal operation and recovery are the same code
path. ``compute_pending_actions`` does not know whether the run is
fresh or resumed; it just looks at what state says exists at each
phase and returns dispatches for what's missing or in-flight.

A ``Dispatch`` returned with ``is_redispatch=True`` means the
corresponding ``Requested`` event is already in the log; the driver
fires the provider call without re-emitting. ``is_redispatch=False``
means the driver emits the ``Requested`` event before firing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Generic, Literal, Union

from pydantic import BaseModel, ConfigDict
from ulid import ULID

from arid_badger.hill_climbing.domain import Evaluation, Node, ObservationT
from arid_badger.max_reward_puct.search import (
    calculate_puct_scores,
    select_batch_of_parents,
)
from arid_badger.max_reward_puct.v3.config import SearchConfig
from arid_badger.max_reward_puct.v3.events import (
    CandidateDeferred,
    CandidateSelected,
    EvaluationsDrained,
    ForecastsDrained,
    MutationsDrained,
    SearchEvent,
    StepCompleted,
    StepStarted,
)
from arid_badger.max_reward_puct.v3.state import (
    Candidate,
    CandidateAwaitingEval,
    CandidateAwaitingForecast,
    CandidateAwaitingSelection,
    CandidateEvaluating,
    CandidateForecasting,
    CandidateMutating,
    CandidateSettled,
    ParentInStep,
    ParentPhase,
    SearchState,
)


class _DispatchBase(BaseModel):
    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)


class MutationDispatch(_DispatchBase, Generic[ObservationT]):
    """Run the mutation provider for ``(parent_code, parent_evaluation)``."""

    kind: Literal["mutation"] = "mutation"
    step: int
    request_id: ULID
    parent_ulid: ULID
    parent_code: str
    parent_evaluation: Evaluation[ObservationT]
    is_redispatch: bool


class ForecastDispatch(_DispatchBase):
    """Run the surrogate over ``code``."""

    kind: Literal["forecast"] = "forecast"
    step: int
    request_id: ULID
    parent_ulid: ULID
    code: str
    is_redispatch: bool


class EvaluationDispatch(_DispatchBase):
    """Run the evaluation provider over ``code``."""

    kind: Literal["evaluation"] = "evaluation"
    step: int
    request_id: ULID
    parent_ulid: ULID
    code: str
    is_redispatch: bool


Dispatch = Union[
    MutationDispatch[ObservationT], ForecastDispatch, EvaluationDispatch
]
"""Discriminated union of dispatches. Not serialized — callers pattern-
match on ``kind``. Actions/dispatches live only in process."""


@dataclass
class Actions(Generic[ObservationT]):
    """Output of ``compute_pending_actions``.

    Pure, transient: produced by the algorithm core, consumed by the
    driver loop in the same iteration, never persisted. Two parallel
    bags: events to append before any provider call, plus dispatches
    to fire.

    Order within ``events`` matters for replay: the reducer folds them
    in the order given. ``dispatches`` is order-insensitive.

    A ``dataclass`` (rather than a Pydantic model) because the events
    and dispatches fields use the generic, ``Annotated[Union[...],
    Field(discriminator=...)]`` aliases — Python's typing machinery
    refuses to re-subscribe these aliases at Pydantic field-evaluation
    time, and Actions is never serialized anyway.
    """

    events: list[SearchEvent[ObservationT]] = field(default_factory=list)
    dispatches: list[Dispatch[ObservationT]] = field(default_factory=list)

    def is_empty(self) -> bool:
        return not self.events and not self.dispatches


# --- The pure decision function ----------------------------------------


def compute_pending_actions(
    state: SearchState[ObservationT],
    config: SearchConfig,
) -> Actions[ObservationT]:
    """Decide what to do next based on the current state.

    Pure. The same ``(state, config)`` always produces the same
    ``Actions``. No clock, no randomness, no I/O.
    """
    actions: Actions[ObservationT] = Actions()

    # Step boundary: open a fresh step if budget allows.
    if not state.current_step_active:
        if state.current_step >= config.total_budget_steps:
            return actions
        parents = select_batch_of_parents(
            archive=state.archive,
            batch_size=config.batch_size,
            visit_counts=state.visit_counts,
            best_child_rewards=state.best_child_rewards,
            global_expansion_count=state.global_expansion_count,
            seed_ids=set(state.seed_ids),
            c_puct=config.c_puct,
        )
        if not parents:
            return actions
        # Recover the PUCT score that drove each selection. Pinned in
        # ``StepStarted`` so analyses can read the ranking the
        # algorithm used without re-deriving it.
        all_scored = calculate_puct_scores(
            archive=state.archive,
            visit_counts=state.visit_counts,
            best_child_rewards=state.best_child_rewards,
            global_expansion_count=state.global_expansion_count,
            seed_ids=set(state.seed_ids),
            c_puct=config.c_puct,
        )
        score_by_ulid = {node.ulid: score for score, _reward, node in all_scored}
        actions.events.append(
            StepStarted(
                step=state.current_step,
                parent_ulids=[p.ulid for p in parents],
                selected_parent_scores=[score_by_ulid[p.ulid] for p in parents],
            )
        )
        return actions

    archive_by_ulid = {n.ulid: n for n in state.archive}

    for parent_record in state.current_step_parents:
        parent_node = archive_by_ulid.get(parent_record.parent_ulid)
        if parent_node is None:
            # Parent was evicted from the archive. The phase machine
            # has nothing to do for it; ``_finalize_step`` will skip
            # any orphaned candidates at ``StepCompleted``.
            continue
        match parent_record.phase:
            case ParentPhase.MUTATING_FORECASTING:
                _handle_mutating_forecasting(
                    state, config, parent_node, parent_record, actions
                )
            case ParentPhase.AWAITING_SELECTION:
                _handle_awaiting_selection(
                    state, config, parent_record, actions
                )
            case ParentPhase.EVALUATING:
                _handle_evaluating(state, parent_record, actions)
            case ParentPhase.DONE:
                pass

    # Step completion fence.
    if all(
        p.phase == ParentPhase.DONE for p in state.current_step_parents
    ):
        actions.events.append(StepCompleted(step=state.current_step))

    return actions


# --- Phase handlers ----------------------------------------------------


def _handle_mutating_forecasting(
    state: SearchState[ObservationT],
    config: SearchConfig,
    parent_node: Node[ObservationT],
    parent_record: ParentInStep[ObservationT],
    actions: Actions[ObservationT],
) -> None:
    parent_ulid = parent_node.ulid
    candidates = list(parent_record.candidates.values())

    # Re-dispatch in-flight items + advance fresh forecasts.
    for cand in candidates:
        if isinstance(cand, CandidateMutating):
            actions.dispatches.append(
                MutationDispatch(
                    step=cand.step,
                    request_id=cand.request_id,
                    parent_ulid=parent_ulid,
                    parent_code=parent_node.program_code,
                    parent_evaluation=parent_node.evaluation,
                    is_redispatch=True,
                )
            )
        elif isinstance(cand, CandidateAwaitingForecast):
            actions.dispatches.append(
                ForecastDispatch(
                    step=cand.step,
                    request_id=cand.request_id,
                    parent_ulid=parent_ulid,
                    code=cand.code,
                    is_redispatch=False,
                )
            )
        elif isinstance(cand, CandidateForecasting):
            actions.dispatches.append(
                ForecastDispatch(
                    step=cand.step,
                    request_id=cand.request_id,
                    parent_ulid=parent_ulid,
                    code=cand.code,
                    is_redispatch=True,
                )
            )
        # AwaitingSelection / Settled — wait for siblings or already
        # terminal. Nothing to do.

    # Open fresh mutation slots up to ``samples_per_parent``.
    missing = config.samples_per_parent - len(candidates)
    for _ in range(max(0, missing)):
        actions.dispatches.append(
            MutationDispatch(
                step=state.current_step,
                request_id=ULID(),
                parent_ulid=parent_ulid,
                parent_code=parent_node.program_code,
                parent_evaluation=parent_node.evaluation,
                is_redispatch=False,
            )
        )

    # Barrier emissions.
    have_full_slate = len(candidates) >= config.samples_per_parent
    if have_full_slate:
        if all(_mutation_terminated(c) for c in candidates):
            if not parent_record.mutations_drained:
                actions.events.append(
                    MutationsDrained(
                        step=state.current_step, parent_ulid=parent_ulid
                    )
                )
        if all(_forecast_terminated(c) for c in candidates):
            actions.events.append(
                ForecastsDrained(
                    step=state.current_step, parent_ulid=parent_ulid
                )
            )


def _mutation_terminated(cand: Candidate) -> bool:
    """A candidate has crossed the mutation phase iff it has a
    successful follow-on phase or it has settled (mutation_failed
    is the only settled reason that occurs before the mutation phase
    is finished)."""
    return not isinstance(cand, CandidateMutating)


def _forecast_terminated(cand: Candidate) -> bool:
    """A candidate has crossed the forecast phase iff it has reached
    AwaitingSelection (forecast succeeded) or settled with a reason
    that occurs at or before the forecast phase (mutation_failed,
    forecast_failed)."""
    if isinstance(cand, CandidateAwaitingSelection):
        return True
    if isinstance(cand, CandidateSettled):
        return cand.reason in ("mutation_failed", "forecast_failed")
    return False


def _handle_awaiting_selection(
    state: SearchState[ObservationT],
    config: SearchConfig,
    parent_record: ParentInStep[ObservationT],
    actions: Actions[ObservationT],
) -> None:
    """Apply the ranking rule and emit one ``CandidateSelected`` per
    promoted slot, ``CandidateDeferred`` for everyone else.

    Failed forecasts are *not* in the awaiting-selection set — they
    settled directly under ``forecast_failed`` and never enter the
    selection. This preserves the per-parent eval budget invariant of
    exactly ``k_per_parent`` (spec § invariants).
    """
    awaiting = [
        c
        for c in parent_record.candidates.values()
        if isinstance(c, CandidateAwaitingSelection)
    ]
    if not awaiting:
        # Every candidate failed before reaching AwaitingSelection
        # (e.g. all forecasts failed). The eval set is empty by
        # vacuous truth — every selected candidate (zero of them)
        # has a terminal eval outcome — so emit EvaluationsDrained
        # directly to advance phase past EVALUATING straight to
        # DONE. Without this, the parent sticks in AWAITING_SELECTION
        # forever and the driver returns without firing StepCompleted.
        actions.events.append(
            EvaluationsDrained(
                step=state.current_step, parent_ulid=parent_record.parent_ulid
            )
        )
        return

    scored: list[tuple[float, CandidateAwaitingSelection]] = [
        (
            config.ranking_rule.score(
                forecast=c.forecast, archive=state.archive
            ),
            c,
        )
        for c in awaiting
    ]
    # Deterministic tiebreaking: descending score, then request_id.
    # No randomness — spec § principles.
    scored.sort(key=lambda t: (-t[0], t[1].request_id))

    selected = scored[: config.k_per_parent]
    deferred = scored[config.k_per_parent :]

    for score, cand in selected:
        actions.events.append(
            CandidateSelected(
                step=cand.step,
                request_id=cand.request_id,
                parent_ulid=parent_record.parent_ulid,
                selection_score=score,
            )
        )
    for score, cand in deferred:
        actions.events.append(
            CandidateDeferred(
                step=cand.step,
                request_id=cand.request_id,
                parent_ulid=parent_record.parent_ulid,
                selection_score=score,
            )
        )


def _handle_evaluating(
    state: SearchState[ObservationT],
    parent_record: ParentInStep[ObservationT],
    actions: Actions[ObservationT],
) -> None:
    parent_ulid = parent_record.parent_ulid
    candidates = list(parent_record.candidates.values())

    # Eval-relevant candidates: those that were selected (now
    # AwaitingEval, Evaluating, or Settled with reason in
    # {evaluated, eval_failed}).
    evalable = [
        c
        for c in candidates
        if isinstance(c, (CandidateAwaitingEval, CandidateEvaluating))
        or (
            isinstance(c, CandidateSettled)
            and c.reason in ("evaluated", "eval_failed")
        )
    ]

    for cand in candidates:
        if isinstance(cand, CandidateAwaitingEval):
            actions.dispatches.append(
                EvaluationDispatch(
                    step=cand.step,
                    request_id=cand.request_id,
                    parent_ulid=parent_ulid,
                    code=cand.code,
                    is_redispatch=False,
                )
            )
        elif isinstance(cand, CandidateEvaluating):
            actions.dispatches.append(
                EvaluationDispatch(
                    step=cand.step,
                    request_id=cand.request_id,
                    parent_ulid=parent_ulid,
                    code=cand.code,
                    is_redispatch=True,
                )
            )

    # EvaluationsDrained fires when every selected candidate is
    # settled in {evaluated, eval_failed}. If no candidates were
    # selected at all (all forecasts failed), evalable is empty —
    # drain immediately.
    if all(
        isinstance(c, CandidateSettled)
        and c.reason in ("evaluated", "eval_failed")
        for c in evalable
    ):
        actions.events.append(
            EvaluationsDrained(
                step=state.current_step, parent_ulid=parent_ulid
            )
        )


__all__ = [
    "Actions",
    "Dispatch",
    "EvaluationDispatch",
    "ForecastDispatch",
    "MutationDispatch",
    "compute_pending_actions",
]
