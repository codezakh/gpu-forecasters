"""Pure-reducer tests. No I/O, no futures, no threads — events in, state out."""

from ulid import ULID

from gpu_forecasters.hill_climbing.domain import Evaluation, NoFeedback, Node
from gpu_forecasters.max_reward_puct.v2.events import (
    EvaluationCompleted,
    EvaluationFailed,
    EvaluationRequested,
    MutationCompleted,
    MutationFailed,
    MutationRequested,
    SearchInitialized,
    StepCompleted,
    StepStarted,
)
from gpu_forecasters.max_reward_puct.v2.state import SearchState, apply_event, replay

K = 2
CAP = 1000


def _eval(reward: float | None) -> Evaluation[NoFeedback]:
    return Evaluation(observation=NoFeedback(), reward=reward)


def _root(code: str = "0000", reward: float = 0.0) -> Node[NoFeedback]:
    return Node[NoFeedback](
        program_code=code, evaluation=_eval(reward), ancestors=[], is_seed=True
    )


def _replay(events):
    return replay(
        events, k_per_parent=K, archive_capacity=CAP, observation_type=NoFeedback
    )


def test_search_initialized_sets_root_and_seeds():
    root = _root()
    state: SearchState[NoFeedback] = SearchState()
    _ = apply_event(
        state,
        SearchInitialized[NoFeedback](root=root),
        k_per_parent=K,
        archive_capacity=CAP,
        observation_type=NoFeedback,
    )

    assert state.archive == [root]
    assert state.seed_ids == {root.ulid}
    assert state.current_step == 0


def test_step_happy_path_folds_into_archive():
    root = _root(reward=0.0)
    child_ulid = ULID()
    events = [
        SearchInitialized[NoFeedback](root=root),
        StepStarted(step=0, parent_ulids=[root.ulid]),
        MutationRequested(request_id="m0", parent_ulid=root.ulid),
        MutationCompleted(request_id="m0", code="0001"),
        EvaluationRequested(
            request_id="e0", child_ulid=child_ulid, parent_ulid=root.ulid, code="0001"
        ),
        EvaluationCompleted[NoFeedback](request_id="e0", evaluation=_eval(1.0)),
        StepCompleted(step=0),
    ]
    state = _replay(events)

    assert state.current_step == 1
    assert len(state.archive) == 2
    child = next(n for n in state.archive if n.ulid == child_ulid)
    assert child.program_code == "0001"
    assert child.evaluation.reward == 1.0
    assert child.ancestors == [root.ulid]
    assert state.visit_counts[root.ulid] == 1
    assert state.best_child_rewards[root.ulid] == 1.0
    assert state.global_expansion_count == 1
    assert state.in_flight_children == {}
    assert state.in_flight_mutations == {}


def test_mutation_requested_tracked_for_redispatch():
    """The reducer must surface un-terminated MutationRequested entries
    in ``in_flight_mutations`` so crash recovery can re-dispatch them."""
    root = _root()
    events = [
        SearchInitialized[NoFeedback](root=root),
        StepStarted(step=0, parent_ulids=[root.ulid]),
        MutationRequested(request_id="m0", parent_ulid=root.ulid),
        MutationRequested(request_id="m1", parent_ulid=root.ulid),
        # Only one terminal — m1 is still in flight.
        MutationCompleted(request_id="m0", code="0001"),
    ]
    state = _replay(events)

    # m0 was completed AND the eval was never dispatched: code-bearing
    # entry survives so recovery can dispatch the eval.
    assert "m0" in state.in_flight_mutations
    assert state.in_flight_mutations["m0"].code == "0001"
    # m1 has no completion yet: code is None, recovery will resubmit.
    assert state.in_flight_mutations["m1"].code is None
    assert state.in_flight_mutations["m1"].parent_ulid == root.ulid


def test_eval_requested_with_link_drains_mutation_inflight():
    """The chain MutationRequested → MutationCompleted → EvaluationRequested
    must drain the mutation's in-flight entry once the linked
    EvaluationRequested lands. Otherwise the same work would be re-dispatched
    on recovery as if the eval had never been issued."""
    root = _root()
    c1 = ULID()
    events = [
        SearchInitialized[NoFeedback](root=root),
        StepStarted(step=0, parent_ulids=[root.ulid]),
        MutationRequested(request_id="m0", parent_ulid=root.ulid),
        MutationCompleted(request_id="m0", code="0001"),
        EvaluationRequested(
            request_id="e0",
            child_ulid=c1,
            parent_ulid=root.ulid,
            code="0001",
            from_mutation_request_id="m0",
        ),
    ]
    state = _replay(events)
    assert state.in_flight_mutations == {}
    assert "e0" in state.in_flight_children


def test_evaluation_pending_visible_for_redispatch():
    root = _root()
    c1 = ULID()
    events = [
        SearchInitialized[NoFeedback](root=root),
        StepStarted(step=0, parent_ulids=[root.ulid]),
        MutationRequested(request_id="m0", parent_ulid=root.ulid),
        MutationCompleted(request_id="m0", code="0001"),
        EvaluationRequested(
            request_id="e0", child_ulid=c1, parent_ulid=root.ulid, code="0001"
        ),
        # No terminal.
    ]
    state = _replay(events)
    pending = state.in_flight_children["e0"]
    assert pending.evaluation is None
    assert pending.failed is False


def test_mutation_failed_triggers_failed_rollout():
    root = _root()
    events = [
        SearchInitialized[NoFeedback](root=root),
        StepStarted(step=0, parent_ulids=[root.ulid]),
        MutationRequested(request_id="m0", parent_ulid=root.ulid),
        MutationFailed(request_id="m0", reason="boom"),
        StepCompleted(step=0),
    ]
    state = _replay(events)

    assert len(state.archive) == 1
    assert state.visit_counts[root.ulid] == 1
    assert state.global_expansion_count == 1


def test_evaluation_failed_counts_as_no_valid_child():
    root = _root()
    c1 = ULID()
    events = [
        SearchInitialized[NoFeedback](root=root),
        StepStarted(step=0, parent_ulids=[root.ulid]),
        MutationRequested(request_id="m0", parent_ulid=root.ulid),
        MutationCompleted(request_id="m0", code="0001"),
        EvaluationRequested(
            request_id="e0", child_ulid=c1, parent_ulid=root.ulid, code="0001"
        ),
        EvaluationFailed(request_id="e0", reason="modal timeout"),
        StepCompleted(step=0),
    ]
    state = _replay(events)

    assert len(state.archive) == 1
    assert state.visit_counts[root.ulid] == 1
    assert state.global_expansion_count == 1


def test_none_reward_does_not_count_as_valid_child():
    root = _root()
    c1 = ULID()
    events = [
        SearchInitialized[NoFeedback](root=root),
        StepStarted(step=0, parent_ulids=[root.ulid]),
        MutationRequested(request_id="m0", parent_ulid=root.ulid),
        MutationCompleted(request_id="m0", code="0001"),
        EvaluationRequested(
            request_id="e0", child_ulid=c1, parent_ulid=root.ulid, code="0001"
        ),
        EvaluationCompleted[NoFeedback](request_id="e0", evaluation=_eval(None)),
        StepCompleted(step=0),
    ]
    state = _replay(events)

    assert len(state.archive) == 1
    assert state.visit_counts[root.ulid] == 1


def test_replay_is_deterministic():
    """Same log → same state. The core property for resumability."""
    root = _root()
    c1, c2 = ULID(), ULID()
    events = [
        SearchInitialized[NoFeedback](root=root),
        StepStarted(step=0, parent_ulids=[root.ulid]),
        MutationRequested(request_id="m0", parent_ulid=root.ulid),
        MutationRequested(request_id="m1", parent_ulid=root.ulid),
        MutationCompleted(request_id="m0", code="0001"),
        MutationCompleted(request_id="m1", code="0010"),
        EvaluationRequested(
            request_id="e0", child_ulid=c1, parent_ulid=root.ulid, code="0001"
        ),
        EvaluationRequested(
            request_id="e1", child_ulid=c2, parent_ulid=root.ulid, code="0010"
        ),
        EvaluationCompleted[NoFeedback](request_id="e0", evaluation=_eval(1.0)),
        EvaluationCompleted[NoFeedback](request_id="e1", evaluation=_eval(2.0)),
        StepCompleted(step=0),
    ]
    s1 = _replay(events)
    s2 = _replay(events)

    assert s1.current_step == s2.current_step
    assert {n.ulid for n in s1.archive} == {n.ulid for n in s2.archive}
    assert s1.visit_counts == s2.visit_counts
    assert s1.best_child_rewards == s2.best_child_rewards
    assert s1.global_expansion_count == s2.global_expansion_count


def test_truncated_midstep_leaves_in_flight_visible():
    """Crash mid-step: the reducer surfaces every kind of in-flight
    work so the driver can re-dispatch on recovery."""
    root = _root()
    c1 = ULID()
    events = [
        SearchInitialized[NoFeedback](root=root),
        StepStarted(step=0, parent_ulids=[root.ulid]),
        MutationRequested(request_id="m0", parent_ulid=root.ulid),
        MutationRequested(request_id="m1", parent_ulid=root.ulid),
        MutationCompleted(request_id="m0", code="0001"),
        EvaluationRequested(
            request_id="e0",
            child_ulid=c1,
            parent_ulid=root.ulid,
            code="0001",
            from_mutation_request_id="m0",
        ),
        # Crash here: m1 still pending mutation; e0 pending eval.
    ]
    state = _replay(events)

    assert state.current_step == 0
    assert state.step_parent_ulids == [root.ulid]
    # m0 was drained by the linked EvaluationRequested.
    assert "m0" not in state.in_flight_mutations
    # m1 pending mutation submit.
    assert state.in_flight_mutations["m1"].parent_ulid == root.ulid
    assert state.in_flight_mutations["m1"].code is None
    # e0 pending eval.
    assert "e0" in state.in_flight_children
    assert state.in_flight_children["e0"].evaluation is None
    assert not state.in_flight_children["e0"].failed
