"""Pure-decision tests for v3 ``compute_pending_actions``.

Synthetic states cover: fresh start, mid-step pre-forecasts-drained,
post-forecasts-drained, mid-evaluation, end-of-step (spec § acceptance
criteria).
"""

from __future__ import annotations

from ulid import ULID

from arid_badger.hill_climbing.domain import Evaluation, NoFeedback, Node
from arid_badger.landscape_map.v2 import (
    SUCCESS_BINS,
    KernelRuntimeEstimate,
    SpeedupBin,
)
from arid_badger.max_reward_puct.v3.actions import (
    EvaluationDispatch,
    ForecastDispatch,
    MutationDispatch,
    compute_pending_actions,
)
from arid_badger.max_reward_puct.v3.config import (
    ExpectedBinIndexRule,
    SearchConfig,
)
from arid_badger.max_reward_puct.v3.events import (
    CandidateDeferred,
    CandidateSelected,
    EvaluationsDrained,
    ForecastsDrained,
    MutationsDrained,
    StepCompleted,
    StepStarted,
)
from arid_badger.max_reward_puct.v3.state import (
    CandidateAwaitingEval,
    CandidateAwaitingForecast,
    CandidateAwaitingSelection,
    CandidateEvaluating,
    CandidateMutating,
    CandidateSettled,
    ParentInStep,
    ParentPhase,
    SearchState,
)


def _eval(reward: float | None) -> Evaluation[NoFeedback]:
    return Evaluation(observation=NoFeedback(), reward=reward)


def _root() -> Node[NoFeedback]:
    return Node[NoFeedback](
        program_code="0000",
        evaluation=_eval(0.0),
        ancestors=[],
        is_seed=True,
    )


def _config(
    *, total_budget_steps: int = 5, samples_per_parent: int = 3, k_per_parent: int = 2
) -> SearchConfig:
    return SearchConfig(
        total_budget_steps=total_budget_steps,
        batch_size=1,
        samples_per_parent=samples_per_parent,
        k_per_parent=k_per_parent,
        ranking_rule=ExpectedBinIndexRule(),
    )


def _uniform_estimate(predicted: SpeedupBin = SpeedupBin.MINOR_SLOWDOWN) -> KernelRuntimeEstimate:
    spread = 1.0 / len(SUCCESS_BINS)
    return KernelRuntimeEstimate(
        predicted_bin=predicted,
        bin_probabilities={b: spread for b in SUCCESS_BINS},
        reasoning="test",
        raw_probability_sum=1.0,
    )


def test_fresh_start_emits_step_started_only():
    """No active step; budget allows: emit StepStarted, no dispatches yet."""
    root = _root()
    state = SearchState[NoFeedback](
        archive=(root,),
        seed_ids=frozenset({root.ulid}),
        current_step=0,
    )
    actions = compute_pending_actions(state, _config())
    assert len(actions.events) == 1
    assert isinstance(actions.events[0], StepStarted)
    assert actions.events[0].parent_ulids == [root.ulid]
    assert not actions.dispatches


def test_budget_exhausted_emits_nothing():
    root = _root()
    state = SearchState[NoFeedback](
        archive=(root,),
        seed_ids=frozenset({root.ulid}),
        current_step=5,
    )
    actions = compute_pending_actions(state, _config(total_budget_steps=5))
    assert actions.is_empty()


def test_mid_step_dispatches_fresh_mutations_for_missing_slots():
    """Step active, parent has no candidates yet: returns
    samples_per_parent fresh mutation dispatches."""
    root = _root()
    parent = ParentInStep[NoFeedback](
        parent_ulid=root.ulid, phase=ParentPhase.MUTATING_FORECASTING
    )
    state = SearchState[NoFeedback](
        archive=(root,),
        seed_ids=frozenset({root.ulid}),
        current_step=0,
        current_step_parents=(parent,),
    )
    config = _config(samples_per_parent=3)
    actions = compute_pending_actions(state, config)
    assert all(isinstance(d, MutationDispatch) for d in actions.dispatches)
    assert len(actions.dispatches) == 3
    assert all(not d.is_redispatch for d in actions.dispatches)
    assert not actions.events


def test_mid_step_redispatches_in_flight_mutation():
    """Mutation already requested but not completed: returns a
    redispatch with the same request_id."""
    root = _root()
    req = ULID()
    parent = ParentInStep[NoFeedback](
        parent_ulid=root.ulid,
        phase=ParentPhase.MUTATING_FORECASTING,
        candidates={
            req: CandidateMutating(step=0, request_id=req, parent_ulid=root.ulid)
        },
    )
    state = SearchState[NoFeedback](
        archive=(root,),
        seed_ids=frozenset({root.ulid}),
        current_step=0,
        current_step_parents=(parent,),
    )
    config = _config(samples_per_parent=1)
    actions = compute_pending_actions(state, config)
    assert len(actions.dispatches) == 1
    d = actions.dispatches[0]
    assert isinstance(d, MutationDispatch)
    assert d.request_id == req
    assert d.is_redispatch is True


def test_mutation_completed_yields_fresh_forecast_dispatch():
    """Mutation done, forecast not yet requested: returns a fresh
    forecast dispatch."""
    root = _root()
    req = ULID()
    parent = ParentInStep[NoFeedback](
        parent_ulid=root.ulid,
        phase=ParentPhase.MUTATING_FORECASTING,
        candidates={
            req: CandidateAwaitingForecast(
                step=0, request_id=req, parent_ulid=root.ulid, code="0001"
            )
        },
    )
    state = SearchState[NoFeedback](
        archive=(root,),
        seed_ids=frozenset({root.ulid}),
        current_step=0,
        current_step_parents=(parent,),
    )
    actions = compute_pending_actions(state, _config(samples_per_parent=1))
    assert any(isinstance(d, ForecastDispatch) for d in actions.dispatches)
    forecast_dispatches = [
        d for d in actions.dispatches if isinstance(d, ForecastDispatch)
    ]
    assert len(forecast_dispatches) == 1
    assert forecast_dispatches[0].is_redispatch is False
    assert forecast_dispatches[0].code == "0001"


def test_all_forecasts_done_emits_forecasts_drained():
    """Full slate of candidates, all in AwaitingSelection: emit
    ForecastsDrained barrier."""
    root = _root()
    candidate_ids = [ULID() for _ in range(2)]
    parent = ParentInStep[NoFeedback](
        parent_ulid=root.ulid,
        phase=ParentPhase.MUTATING_FORECASTING,
        mutations_drained=True,  # already emitted on a prior iteration
        candidates={
            rid: CandidateAwaitingSelection(
                step=0,
                request_id=rid,
                parent_ulid=root.ulid,
                code=f"000{i}",
                forecast=_uniform_estimate(),
            )
            for i, rid in enumerate(candidate_ids)
        },
    )
    state = SearchState[NoFeedback](
        archive=(root,),
        seed_ids=frozenset({root.ulid}),
        current_step=0,
        current_step_parents=(parent,),
    )
    config = _config(samples_per_parent=2)
    actions = compute_pending_actions(state, config)
    drained = [e for e in actions.events if isinstance(e, ForecastsDrained)]
    assert len(drained) == 1


def test_mutations_drained_only_emitted_once():
    """Once mutations_drained is True on the parent record, the
    decision function does not re-emit MutationsDrained."""
    root = _root()
    rid = ULID()
    parent = ParentInStep[NoFeedback](
        parent_ulid=root.ulid,
        phase=ParentPhase.MUTATING_FORECASTING,
        mutations_drained=True,
        candidates={
            rid: CandidateAwaitingForecast(
                step=0,
                request_id=rid,
                parent_ulid=root.ulid,
                code="0001",
            ),
        },
    )
    state = SearchState[NoFeedback](
        archive=(root,),
        seed_ids=frozenset({root.ulid}),
        current_step=0,
        current_step_parents=(parent,),
    )
    actions = compute_pending_actions(state, _config(samples_per_parent=1))
    assert not any(isinstance(e, MutationsDrained) for e in actions.events)


def test_awaiting_selection_emits_top_k_selected_and_rest_deferred():
    """In AwaitingSelection: the ranking rule picks top-k; the rest get
    Deferred. CandidateSettled(forecast_failed) does not enter the
    selection set (preserves the per-parent eval budget invariant)."""
    root = _root()
    candidates = {}
    # 3 awaiting-selection candidates with distinct predicted bins so
    # ExpectedBinIndexRule produces a strict ordering.
    rids: list[ULID] = []
    for i, predicted in enumerate(
        (SpeedupBin.HIGH_SPEEDUP, SpeedupBin.MINOR_SPEEDUP, SpeedupBin.SEVERE_SLOWDOWN)
    ):
        rid = ULID()
        rids.append(rid)
        candidates[rid] = CandidateAwaitingSelection(
            step=0,
            request_id=rid,
            parent_ulid=root.ulid,
            code=f"000{i}",
            forecast=KernelRuntimeEstimate(
                predicted_bin=predicted,
                bin_probabilities={
                    b: (0.9 if b == predicted else 0.1 / (len(SUCCESS_BINS) - 1))
                    for b in SUCCESS_BINS
                },
                reasoning="t",
                raw_probability_sum=1.0,
            ),
        )
    # And one already settled forecast_failed — should not be selected.
    failed_rid = ULID()
    candidates[failed_rid] = CandidateSettled[NoFeedback](
        step=0,
        request_id=failed_rid,
        parent_ulid=root.ulid,
        reason="forecast_failed",
        code="9999",
    )
    parent = ParentInStep[NoFeedback](
        parent_ulid=root.ulid,
        phase=ParentPhase.AWAITING_SELECTION,
        candidates=candidates,
    )
    state = SearchState[NoFeedback](
        archive=(root,),
        seed_ids=frozenset({root.ulid}),
        current_step=0,
        current_step_parents=(parent,),
    )
    config = _config(samples_per_parent=4, k_per_parent=2)
    actions = compute_pending_actions(state, config)
    selected = [e for e in actions.events if isinstance(e, CandidateSelected)]
    deferred = [e for e in actions.events if isinstance(e, CandidateDeferred)]
    assert len(selected) == 2
    assert len(deferred) == 1  # only the 3rd awaiting one; failed candidate excluded
    selected_ids = {e.request_id for e in selected}
    # The two highest-bin candidates are picked.
    assert rids[0] in selected_ids  # HIGH_SPEEDUP
    assert rids[1] in selected_ids  # MINOR_SPEEDUP


def test_evaluating_phase_dispatches_eval_for_awaiting_eval():
    root = _root()
    rid = ULID()
    parent = ParentInStep[NoFeedback](
        parent_ulid=root.ulid,
        phase=ParentPhase.EVALUATING,
        candidates={
            rid: CandidateAwaitingEval(
                step=0,
                request_id=rid,
                parent_ulid=root.ulid,
                code="0001",
                forecast=_uniform_estimate(),
                selection_score=1.0,
            )
        },
    )
    state = SearchState[NoFeedback](
        archive=(root,),
        seed_ids=frozenset({root.ulid}),
        current_step=0,
        current_step_parents=(parent,),
    )
    actions = compute_pending_actions(state, _config())
    eval_dispatches = [
        d for d in actions.dispatches if isinstance(d, EvaluationDispatch)
    ]
    assert len(eval_dispatches) == 1
    assert eval_dispatches[0].is_redispatch is False


def test_evaluating_phase_redispatches_in_flight_eval():
    root = _root()
    rid = ULID()
    parent = ParentInStep[NoFeedback](
        parent_ulid=root.ulid,
        phase=ParentPhase.EVALUATING,
        candidates={
            rid: CandidateEvaluating(
                step=0,
                request_id=rid,
                parent_ulid=root.ulid,
                code="0001",
                forecast=_uniform_estimate(),
                selection_score=1.0,
            )
        },
    )
    state = SearchState[NoFeedback](
        archive=(root,),
        seed_ids=frozenset({root.ulid}),
        current_step=0,
        current_step_parents=(parent,),
    )
    actions = compute_pending_actions(state, _config())
    eval_dispatches = [
        d for d in actions.dispatches if isinstance(d, EvaluationDispatch)
    ]
    assert len(eval_dispatches) == 1
    assert eval_dispatches[0].is_redispatch is True


def test_evaluating_phase_emits_evaluations_drained_when_all_settled():
    root = _root()
    rid = ULID()
    parent = ParentInStep[NoFeedback](
        parent_ulid=root.ulid,
        phase=ParentPhase.EVALUATING,
        candidates={
            rid: CandidateSettled[NoFeedback](
                step=0,
                request_id=rid,
                parent_ulid=root.ulid,
                reason="evaluated",
                code="0001",
                forecast=_uniform_estimate(),
                selection_score=1.0,
                evaluation=_eval(1.0),
            )
        },
    )
    state = SearchState[NoFeedback](
        archive=(root,),
        seed_ids=frozenset({root.ulid}),
        current_step=0,
        current_step_parents=(parent,),
    )
    actions = compute_pending_actions(state, _config())
    drained = [e for e in actions.events if isinstance(e, EvaluationsDrained)]
    assert len(drained) == 1


def test_awaiting_selection_with_no_candidates_drains_immediately():
    """All forecasts failed: every candidate is settled with
    reason=forecast_failed, so awaiting-selection set is empty. The
    decision function must short-circuit by emitting EvaluationsDrained
    so the parent advances to DONE — otherwise the step never completes
    and the driver returns without firing StepCompleted. This is the
    bug the live-Modal smoke caught."""
    root = _root()
    candidates = {}
    for i in range(2):
        rid = ULID()
        candidates[rid] = CandidateSettled[NoFeedback](
            step=0,
            request_id=rid,
            parent_ulid=root.ulid,
            reason="forecast_failed",
            code=f"000{i}",
        )
    parent = ParentInStep[NoFeedback](
        parent_ulid=root.ulid,
        phase=ParentPhase.AWAITING_SELECTION,
        candidates=candidates,
    )
    state = SearchState[NoFeedback](
        archive=(root,),
        seed_ids=frozenset({root.ulid}),
        current_step=0,
        current_step_parents=(parent,),
    )
    actions = compute_pending_actions(state, _config(samples_per_parent=2, k_per_parent=1))
    selected = [e for e in actions.events if isinstance(e, CandidateSelected)]
    deferred = [e for e in actions.events if isinstance(e, CandidateDeferred)]
    drained = [e for e in actions.events if isinstance(e, EvaluationsDrained)]
    assert not selected
    assert not deferred
    assert len(drained) == 1
    assert drained[0].parent_ulid == root.ulid


def test_all_parents_done_emits_step_completed():
    root = _root()
    parent = ParentInStep[NoFeedback](
        parent_ulid=root.ulid, phase=ParentPhase.DONE
    )
    state = SearchState[NoFeedback](
        archive=(root,),
        seed_ids=frozenset({root.ulid}),
        current_step=0,
        current_step_parents=(parent,),
    )
    actions = compute_pending_actions(state, _config())
    completes = [e for e in actions.events if isinstance(e, StepCompleted)]
    assert len(completes) == 1
    assert completes[0].step == 0


