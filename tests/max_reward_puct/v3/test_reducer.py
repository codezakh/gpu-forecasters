"""Pure-reducer tests for v3.

One assertion per event type, on minimal state inputs (spec §
acceptance criteria). No I/O, no futures, no threads — events in,
state out, ``apply_event`` returns a new ``SearchState``.
"""

from __future__ import annotations

import pytest
from ulid import ULID

from arid_badger.hill_climbing.domain import Evaluation, NoFeedback, Node
from arid_badger.landscape_map.v2 import (
    SUCCESS_BINS,
    HardwareContext,
    KernelRuntimeEstimate,
    KernelTaskInfo,
    SpeedupBin,
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
    SearchInitialized,
    StepCompleted,
    StepStarted,
)
from arid_badger.max_reward_puct.v3.state import (
    CandidateAwaitingEval,
    CandidateAwaitingForecast,
    CandidateAwaitingSelection,
    CandidateEvaluating,
    CandidateForecasting,
    CandidateMutating,
    CandidateSettled,
    ParentPhase,
    ReducerStateError,
    SearchState,
    apply_event,
    replay,
)


K = 2
CAP = 1000

_TEST_TASK = KernelTaskInfo(op_name="t", level_id=0, task_id=0)
_TEST_HARDWARE = HardwareContext(
    device_name="test-cpu",
    compute_capability=(0, 0),
    total_global_memory_gb=0.0,
    multiprocessor_count=0,
    max_threads_per_multiprocessor=0,
    clock_rate_ghz=0.0,
    memory_clock_rate_ghz=0.0,
    memory_bus_width_bits=0,
)


def _init_event(root: Node[NoFeedback]) -> SearchInitialized[NoFeedback]:
    return SearchInitialized[NoFeedback](
        root=root,
        kernel_task=_TEST_TASK,
        seed_reference_code="0000",
        hardware=_TEST_HARDWARE,
    )


def _eval(reward: float | None) -> Evaluation[NoFeedback]:
    return Evaluation(observation=NoFeedback(), reward=reward)


def _root(code: str = "0000", reward: float = 0.0) -> Node[NoFeedback]:
    return Node[NoFeedback](
        program_code=code, evaluation=_eval(reward), ancestors=[], is_seed=True
    )


def _uniform_estimate() -> KernelRuntimeEstimate:
    spread = 1.0 / len(SUCCESS_BINS)
    return KernelRuntimeEstimate(
        predicted_bin=SpeedupBin.MINOR_SLOWDOWN,
        bin_probabilities={b: spread for b in SUCCESS_BINS},
        reasoning="test",
        raw_probability_sum=1.0,
    )


def _apply(state, event):
    return apply_event(
        state, event, k_per_parent=K, archive_capacity=CAP, observation_type=NoFeedback
    )


def _start_state_with_one_parent(
    *, request_id: str = "REQ"
) -> tuple[SearchState[NoFeedback], Node[NoFeedback]]:
    """Helper: state initialized with a single root parent and a step
    open with that parent. Used by per-candidate-event tests."""
    root = _root()
    state = SearchState[NoFeedback]()
    state = _apply(state, _init_event(root))
    state = _apply(
        state,
        StepStarted(step=0, parent_ulids=[root.ulid], selected_parent_scores=[0.0]),
    )
    return state, root


# --- Per-search/per-step events ---------------------------------------


def test_search_initialized_seeds_archive():
    root = _root()
    state = SearchState[NoFeedback]()
    new_state = _apply(state, _init_event(root))
    assert new_state.archive == (root,)
    assert new_state.seed_ids == frozenset({root.ulid})
    # Reducer is pure: original state untouched.
    assert state.archive == ()


def test_step_started_opens_active_step():
    root = _root()
    state = _apply(SearchState[NoFeedback](), _init_event(root))
    new_state = _apply(
        state,
        StepStarted(step=0, parent_ulids=[root.ulid], selected_parent_scores=[1.0]),
    )
    assert new_state.current_step_active
    assert len(new_state.current_step_parents) == 1
    parent_record = new_state.current_step_parents[0]
    assert parent_record.parent_ulid == root.ulid
    assert parent_record.phase == ParentPhase.MUTATING_FORECASTING
    assert parent_record.mutations_drained is False
    assert dict(parent_record.candidates) == {}


def test_step_completed_finalizes_and_advances_step():
    """End-to-end mini step: SearchInitialized → StepStarted →
    Mutation+Forecast+Selection+Evaluation → StepCompleted folds the
    child into the archive and bumps current_step."""
    state, root = _start_state_with_one_parent()
    req = "01KQY00000000000000000000A"  # valid ULID

    state = _apply(
        state,
        MutationRequested(step=0, request_id=req, parent_ulid=root.ulid),
    )
    state = _apply(
        state,
        MutationCompleted(
            step=0, request_id=req, parent_ulid=root.ulid, code="0001"
        ),
    )
    state = _apply(
        state,
        ForecastRequested(
            step=0, request_id=req, parent_ulid=root.ulid, code="0001"
        ),
    )
    state = _apply(
        state,
        ForecastCompleted(
            step=0,
            request_id=req,
            parent_ulid=root.ulid,
            forecast=_uniform_estimate(),
        ),
    )
    state = _apply(
        state,
        ForecastsDrained(step=0, parent_ulid=root.ulid),
    )
    state = _apply(
        state,
        CandidateSelected(
            step=0, request_id=req, parent_ulid=root.ulid, selection_score=1.0
        ),
    )
    state = _apply(
        state,
        EvaluationRequested(
            step=0, request_id=req, parent_ulid=root.ulid, code="0001"
        ),
    )
    state = _apply(
        state,
        EvaluationCompleted[NoFeedback](
            step=0,
            request_id=req,
            parent_ulid=root.ulid,
            evaluation=_eval(1.0),
        ),
    )
    state = _apply(state, EvaluationsDrained(step=0, parent_ulid=root.ulid))
    state = _apply(state, StepCompleted(step=0))

    assert state.current_step == 1
    assert not state.current_step_active
    # Child folded into archive.
    assert len(state.archive) == 2
    child = next(n for n in state.archive if n.program_code == "0001")
    assert child.evaluation.reward == 1.0
    assert child.ulid == ULID.from_str(req)
    # Parent has visit count + best_child_reward updated.
    assert state.visit_counts[root.ulid] == 1
    assert state.best_child_rewards[root.ulid] == 1.0


# --- Per-candidate lifecycle (one assertion per event type) -----------


def test_mutation_requested_creates_mutating_candidate():
    state, root = _start_state_with_one_parent()
    req = "01KQY00000000000000000000A"
    new_state = _apply(
        state, MutationRequested(step=0, request_id=req, parent_ulid=root.ulid)
    )
    parent = new_state.parent_in_step(root.ulid)
    assert parent is not None
    cand = parent.candidates[req]
    assert isinstance(cand, CandidateMutating)
    assert cand.request_id == req


def test_mutation_completed_advances_to_awaiting_forecast():
    state, root = _start_state_with_one_parent()
    req = "01KQY00000000000000000000A"
    state = _apply(
        state, MutationRequested(step=0, request_id=req, parent_ulid=root.ulid)
    )
    new_state = _apply(
        state,
        MutationCompleted(
            step=0, request_id=req, parent_ulid=root.ulid, code="0001"
        ),
    )
    parent = new_state.parent_in_step(root.ulid)
    assert parent is not None
    cand = parent.candidates[req]
    assert isinstance(cand, CandidateAwaitingForecast)
    assert cand.code == "0001"


def test_mutation_failed_settles_with_mutation_failed_reason():
    state, root = _start_state_with_one_parent()
    req = "01KQY00000000000000000000A"
    state = _apply(
        state, MutationRequested(step=0, request_id=req, parent_ulid=root.ulid)
    )
    new_state = _apply(
        state,
        MutationFailed(
            step=0, request_id=req, parent_ulid=root.ulid, reason="boom"
        ),
    )
    parent = new_state.parent_in_step(root.ulid)
    assert parent is not None
    cand = parent.candidates[req]
    assert isinstance(cand, CandidateSettled)
    assert cand.reason == "mutation_failed"


def test_forecast_requested_advances_to_forecasting():
    state, root = _start_state_with_one_parent()
    req = "01KQY00000000000000000000A"
    state = _apply(
        state, MutationRequested(step=0, request_id=req, parent_ulid=root.ulid)
    )
    state = _apply(
        state,
        MutationCompleted(
            step=0, request_id=req, parent_ulid=root.ulid, code="0001"
        ),
    )
    new_state = _apply(
        state,
        ForecastRequested(
            step=0, request_id=req, parent_ulid=root.ulid, code="0001"
        ),
    )
    parent = new_state.parent_in_step(root.ulid)
    assert parent is not None
    cand = parent.candidates[req]
    assert isinstance(cand, CandidateForecasting)


def test_forecast_completed_advances_to_awaiting_selection():
    state, root = _start_state_with_one_parent()
    req = "01KQY00000000000000000000A"
    state = _apply(
        state, MutationRequested(step=0, request_id=req, parent_ulid=root.ulid)
    )
    state = _apply(
        state,
        MutationCompleted(
            step=0, request_id=req, parent_ulid=root.ulid, code="0001"
        ),
    )
    state = _apply(
        state,
        ForecastRequested(
            step=0, request_id=req, parent_ulid=root.ulid, code="0001"
        ),
    )
    new_state = _apply(
        state,
        ForecastCompleted(
            step=0,
            request_id=req,
            parent_ulid=root.ulid,
            forecast=_uniform_estimate(),
        ),
    )
    parent = new_state.parent_in_step(root.ulid)
    assert parent is not None
    cand = parent.candidates[req]
    assert isinstance(cand, CandidateAwaitingSelection)
    assert cand.forecast.predicted_bin == SpeedupBin.MINOR_SLOWDOWN


def test_forecast_failed_settles():
    state, root = _start_state_with_one_parent()
    req = "01KQY00000000000000000000A"
    state = _apply(
        state, MutationRequested(step=0, request_id=req, parent_ulid=root.ulid)
    )
    state = _apply(
        state,
        MutationCompleted(
            step=0, request_id=req, parent_ulid=root.ulid, code="0001"
        ),
    )
    new_state = _apply(
        state,
        ForecastFailed(
            step=0, request_id=req, parent_ulid=root.ulid, reason="bad"
        ),
    )
    parent = new_state.parent_in_step(root.ulid)
    assert parent is not None
    cand = parent.candidates[req]
    assert isinstance(cand, CandidateSettled)
    assert cand.reason == "forecast_failed"
    assert cand.code == "0001"


def test_candidate_selected_advances_phase_and_creates_awaiting_eval():
    state, root = _start_state_with_one_parent()
    req = "01KQY00000000000000000000A"
    state = _apply(
        state, MutationRequested(step=0, request_id=req, parent_ulid=root.ulid)
    )
    state = _apply(
        state,
        MutationCompleted(
            step=0, request_id=req, parent_ulid=root.ulid, code="0001"
        ),
    )
    state = _apply(
        state,
        ForecastRequested(
            step=0, request_id=req, parent_ulid=root.ulid, code="0001"
        ),
    )
    state = _apply(
        state,
        ForecastCompleted(
            step=0,
            request_id=req,
            parent_ulid=root.ulid,
            forecast=_uniform_estimate(),
        ),
    )
    state = _apply(state, ForecastsDrained(step=0, parent_ulid=root.ulid))
    new_state = _apply(
        state,
        CandidateSelected(
            step=0, request_id=req, parent_ulid=root.ulid, selection_score=1.0
        ),
    )
    parent = new_state.parent_in_step(root.ulid)
    assert parent is not None
    assert parent.phase == ParentPhase.EVALUATING
    cand = parent.candidates[req]
    assert isinstance(cand, CandidateAwaitingEval)


def test_candidate_deferred_settles_and_advances_phase():
    state, root = _start_state_with_one_parent()
    req = "01KQY00000000000000000000A"
    state = _apply(
        state, MutationRequested(step=0, request_id=req, parent_ulid=root.ulid)
    )
    state = _apply(
        state,
        MutationCompleted(
            step=0, request_id=req, parent_ulid=root.ulid, code="0001"
        ),
    )
    state = _apply(
        state,
        ForecastRequested(
            step=0, request_id=req, parent_ulid=root.ulid, code="0001"
        ),
    )
    state = _apply(
        state,
        ForecastCompleted(
            step=0,
            request_id=req,
            parent_ulid=root.ulid,
            forecast=_uniform_estimate(),
        ),
    )
    state = _apply(state, ForecastsDrained(step=0, parent_ulid=root.ulid))
    new_state = _apply(
        state,
        CandidateDeferred(
            step=0, request_id=req, parent_ulid=root.ulid, selection_score=0.5
        ),
    )
    parent = new_state.parent_in_step(root.ulid)
    assert parent is not None
    assert parent.phase == ParentPhase.EVALUATING
    cand = parent.candidates[req]
    assert isinstance(cand, CandidateSettled)
    assert cand.reason == "deferred"


def test_evaluation_requested_advances_to_evaluating():
    state, root = _start_state_with_one_parent()
    req = "01KQY00000000000000000000A"
    state = _apply(
        state, MutationRequested(step=0, request_id=req, parent_ulid=root.ulid)
    )
    state = _apply(
        state,
        MutationCompleted(
            step=0, request_id=req, parent_ulid=root.ulid, code="0001"
        ),
    )
    state = _apply(
        state,
        ForecastRequested(
            step=0, request_id=req, parent_ulid=root.ulid, code="0001"
        ),
    )
    state = _apply(
        state,
        ForecastCompleted(
            step=0,
            request_id=req,
            parent_ulid=root.ulid,
            forecast=_uniform_estimate(),
        ),
    )
    state = _apply(state, ForecastsDrained(step=0, parent_ulid=root.ulid))
    state = _apply(
        state,
        CandidateSelected(
            step=0, request_id=req, parent_ulid=root.ulid, selection_score=1.0
        ),
    )
    new_state = _apply(
        state,
        EvaluationRequested(
            step=0, request_id=req, parent_ulid=root.ulid, code="0001"
        ),
    )
    parent = new_state.parent_in_step(root.ulid)
    assert parent is not None
    cand = parent.candidates[req]
    assert isinstance(cand, CandidateEvaluating)


def test_evaluation_completed_settles_with_evaluated_reason():
    """Sequence to evaluated terminal."""
    state, root = _start_state_with_one_parent()
    req = "01KQY00000000000000000000A"
    state = _apply(
        state, MutationRequested(step=0, request_id=req, parent_ulid=root.ulid)
    )
    state = _apply(
        state,
        MutationCompleted(
            step=0, request_id=req, parent_ulid=root.ulid, code="0001"
        ),
    )
    state = _apply(
        state,
        ForecastRequested(
            step=0, request_id=req, parent_ulid=root.ulid, code="0001"
        ),
    )
    state = _apply(
        state,
        ForecastCompleted(
            step=0,
            request_id=req,
            parent_ulid=root.ulid,
            forecast=_uniform_estimate(),
        ),
    )
    state = _apply(state, ForecastsDrained(step=0, parent_ulid=root.ulid))
    state = _apply(
        state,
        CandidateSelected(
            step=0, request_id=req, parent_ulid=root.ulid, selection_score=1.0
        ),
    )
    state = _apply(
        state,
        EvaluationRequested(
            step=0, request_id=req, parent_ulid=root.ulid, code="0001"
        ),
    )
    new_state = _apply(
        state,
        EvaluationCompleted[NoFeedback](
            step=0, request_id=req, parent_ulid=root.ulid, evaluation=_eval(1.0)
        ),
    )
    parent = new_state.parent_in_step(root.ulid)
    assert parent is not None
    cand = parent.candidates[req]
    assert isinstance(cand, CandidateSettled)
    assert cand.reason == "evaluated"
    assert cand.evaluation is not None and cand.evaluation.reward == 1.0


def test_evaluation_failed_settles_with_eval_failed_reason():
    state, root = _start_state_with_one_parent()
    req = "01KQY00000000000000000000A"
    state = _apply(
        state, MutationRequested(step=0, request_id=req, parent_ulid=root.ulid)
    )
    state = _apply(
        state,
        MutationCompleted(
            step=0, request_id=req, parent_ulid=root.ulid, code="0001"
        ),
    )
    state = _apply(
        state,
        ForecastRequested(
            step=0, request_id=req, parent_ulid=root.ulid, code="0001"
        ),
    )
    state = _apply(
        state,
        ForecastCompleted(
            step=0,
            request_id=req,
            parent_ulid=root.ulid,
            forecast=_uniform_estimate(),
        ),
    )
    state = _apply(state, ForecastsDrained(step=0, parent_ulid=root.ulid))
    state = _apply(
        state,
        CandidateSelected(
            step=0, request_id=req, parent_ulid=root.ulid, selection_score=1.0
        ),
    )
    state = _apply(
        state,
        EvaluationRequested(
            step=0, request_id=req, parent_ulid=root.ulid, code="0001"
        ),
    )
    new_state = _apply(
        state,
        EvaluationFailed(
            step=0, request_id=req, parent_ulid=root.ulid, reason="oom"
        ),
    )
    parent = new_state.parent_in_step(root.ulid)
    assert parent is not None
    cand = parent.candidates[req]
    assert isinstance(cand, CandidateSettled)
    assert cand.reason == "eval_failed"


# --- Per-parent barriers ----------------------------------------------


def test_mutations_drained_marker_is_recorded():
    state, root = _start_state_with_one_parent()
    new_state = _apply(state, MutationsDrained(step=0, parent_ulid=root.ulid))
    parent = new_state.parent_in_step(root.ulid)
    assert parent is not None
    assert parent.mutations_drained is True
    # phase unchanged — MutationsDrained is a marker.
    assert parent.phase == ParentPhase.MUTATING_FORECASTING


def test_forecasts_drained_advances_phase_to_awaiting_selection():
    state, root = _start_state_with_one_parent()
    new_state = _apply(state, ForecastsDrained(step=0, parent_ulid=root.ulid))
    parent = new_state.parent_in_step(root.ulid)
    assert parent is not None
    assert parent.phase == ParentPhase.AWAITING_SELECTION


def test_evaluations_drained_advances_phase_to_done():
    state, root = _start_state_with_one_parent()
    new_state = _apply(state, EvaluationsDrained(step=0, parent_ulid=root.ulid))
    parent = new_state.parent_in_step(root.ulid)
    assert parent is not None
    assert parent.phase == ParentPhase.DONE


# --- Illegal transitions raise ReducerStateError -----------------------


def test_event_for_unknown_parent_raises():
    """An event whose parent_ulid is not in the active step is a
    driver-construction bug; the reducer surfaces it as a typed error."""
    state, _root = _start_state_with_one_parent()
    bogus_parent = ULID()
    req = "01KQY00000000000000000000A"
    with pytest.raises(ReducerStateError, match="MutationRequested"):
        _apply(
            state,
            MutationRequested(step=0, request_id=req, parent_ulid=bogus_parent),
        )


def test_eval_completed_for_non_evaluating_candidate_raises():
    """EvaluationCompleted requires the candidate to be in
    CandidateEvaluating. Folding it onto a Mutating candidate is a bug."""
    state, root = _start_state_with_one_parent()
    req = "01KQY00000000000000000000A"
    state = _apply(
        state, MutationRequested(step=0, request_id=req, parent_ulid=root.ulid)
    )
    with pytest.raises(ReducerStateError, match="EvaluationCompleted"):
        _apply(
            state,
            EvaluationCompleted[NoFeedback](
                step=0,
                request_id=req,
                parent_ulid=root.ulid,
                evaluation=_eval(1.0),
            ),
        )


def test_candidate_selected_for_non_awaiting_selection_raises():
    """CandidateSelected requires the candidate to be in
    CandidateAwaitingSelection."""
    state, root = _start_state_with_one_parent()
    req = "01KQY00000000000000000000A"
    state = _apply(
        state, MutationRequested(step=0, request_id=req, parent_ulid=root.ulid)
    )
    with pytest.raises(ReducerStateError, match="CandidateSelected"):
        _apply(
            state,
            CandidateSelected(
                step=0, request_id=req, parent_ulid=root.ulid, selection_score=1.0
            ),
        )


# --- Replay determinism ------------------------------------------------


def test_replay_is_deterministic():
    """Spec § invariants: folding the same event sequence twice yields
    equal SearchState values."""
    root = _root()
    req = "01KQY00000000000000000000A"
    events = [
        _init_event(root),
        StepStarted(step=0, parent_ulids=[root.ulid], selected_parent_scores=[0.0]),
        MutationRequested(step=0, request_id=req, parent_ulid=root.ulid),
        MutationCompleted(
            step=0, request_id=req, parent_ulid=root.ulid, code="0001"
        ),
        ForecastRequested(
            step=0, request_id=req, parent_ulid=root.ulid, code="0001"
        ),
        ForecastCompleted(
            step=0,
            request_id=req,
            parent_ulid=root.ulid,
            forecast=_uniform_estimate(),
        ),
        ForecastsDrained(step=0, parent_ulid=root.ulid),
        CandidateSelected(
            step=0, request_id=req, parent_ulid=root.ulid, selection_score=1.0
        ),
        EvaluationRequested(
            step=0, request_id=req, parent_ulid=root.ulid, code="0001"
        ),
        EvaluationCompleted[NoFeedback](
            step=0, request_id=req, parent_ulid=root.ulid, evaluation=_eval(1.0)
        ),
        EvaluationsDrained(step=0, parent_ulid=root.ulid),
        StepCompleted(step=0),
    ]
    a = replay(
        events, k_per_parent=K, archive_capacity=CAP, observation_type=NoFeedback
    )
    b = replay(
        events, k_per_parent=K, archive_capacity=CAP, observation_type=NoFeedback
    )
    assert a.model_dump() == b.model_dump()
