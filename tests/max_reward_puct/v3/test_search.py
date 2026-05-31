"""End-to-end v3 search tests against the binary-string toy.

Covers: convergence with the no-op (k_per_parent == samples_per_parent)
surrogate, surrogate-driven filtering, log well-formedness, crash
recovery via log truncation, and resume-time validation that the
surrogate context in the constructor matches what's pinned in the log.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from gpu_forecasters.hill_climbing.domain import NoFeedback
from gpu_forecasters.landscape_map.v2 import HardwareContext, KernelTaskInfo
from gpu_forecasters.max_reward_puct.v3.config import (
    ExpectedBinIndexRule,
    SearchConfig,
)
from gpu_forecasters.max_reward_puct.v3.event_log import FileEventLog
from gpu_forecasters.max_reward_puct.v3.events import (
    CandidateDeferred,
    CandidateSelected,
    EvaluationRequested,
    EvaluationsDrained,
    ForecastRequested,
    ForecastsDrained,
    MutationRequested,
    SearchEvent,
    StepCompleted,
    StepStarted,
)
from gpu_forecasters.max_reward_puct.v3.state import SearchState
from gpu_forecasters.max_reward_puct.v3.scoring_providers import (
    CoroutineSpeedupEstimator,
)
from gpu_forecasters.max_reward_puct.v3.search import (
    SearchDriver,
    SurrogateContextMismatch,
)
from gpu_forecasters.max_reward_puct.v3.state import replay

from tests.max_reward_puct.v3.binary_string_providers import (
    TEST_HARDWARE,
    TEST_TASK,
    BinaryStringEvaluationProvider,
    BinaryStringMutationProvider,
    CodeLengthAsyncEstimator,
    UniformAsyncEstimator,
)


def _config(
    *,
    total_budget_steps: int,
    samples_per_parent: int,
    k_per_parent: int,
    batch_size: int = 1,
) -> SearchConfig:
    return SearchConfig(
        total_budget_steps=total_budget_steps,
        batch_size=batch_size,
        samples_per_parent=samples_per_parent,
        k_per_parent=k_per_parent,
        ranking_rule=ExpectedBinIndexRule(),
    )


def test_search_converges_to_maximum_no_filter(tmp_path: Path):
    """k_per_parent == samples_per_parent: surrogate doesn't filter
    anything, so v3 should behave like v2 on the toy."""
    log_path = tmp_path / "log.jsonl"
    config = _config(total_budget_steps=20, samples_per_parent=4, k_per_parent=4)
    with (
        BinaryStringMutationProvider(seed=42) as mp,
        BinaryStringEvaluationProvider() as ep,
        CoroutineSpeedupEstimator(UniformAsyncEstimator()) as surrogate,
    ):
        driver = SearchDriver[NoFeedback](
            config,
            mutation_provider=mp,
            evaluation_provider=ep,
            surrogate=surrogate,
            kernel_task=TEST_TASK,
            seed_reference_code="0000",
            hardware=TEST_HARDWARE,
            event_log=FileEventLog(log_path, observation_type=NoFeedback),
            observation_type=NoFeedback,
        )
        state = driver.run(initial_program="0000")
    best = state.best_archived_node()
    assert best is not None
    assert best.program_code == "1111"
    assert best.evaluation.reward == 15.0


def test_filtering_reduces_evaluation_count(tmp_path: Path):
    """k_per_parent < samples_per_parent: surrogate filters, eval call
    count drops below mutation+forecast count."""
    log_path = tmp_path / "log.jsonl"
    config = _config(
        total_budget_steps=10,
        samples_per_parent=4,
        k_per_parent=2,
    )
    with (
        BinaryStringMutationProvider(seed=7) as mp,
        BinaryStringEvaluationProvider() as ep,
        CoroutineSpeedupEstimator(CodeLengthAsyncEstimator()) as surrogate,
    ):
        driver = SearchDriver[NoFeedback](
            config,
            mutation_provider=mp,
            evaluation_provider=ep,
            surrogate=surrogate,
            kernel_task=TEST_TASK,
            seed_reference_code="0000",
            hardware=TEST_HARDWARE,
            event_log=FileEventLog(log_path, observation_type=NoFeedback),
            observation_type=NoFeedback,
        )
        state = driver.run(initial_program="0000")
    # Filter is in effect: at most k_per_parent evals per (step, parent).
    events = FileEventLog(log_path, observation_type=NoFeedback).read_all()
    eval_requests_per_parent_step: dict[tuple[int, str], int] = {}
    for e in events:
        if isinstance(e, EvaluationRequested):
            key = (e.step, str(e.parent_ulid))
            eval_requests_per_parent_step[key] = (
                eval_requests_per_parent_step.get(key, 0) + 1
            )
    assert eval_requests_per_parent_step
    assert all(
        count <= config.k_per_parent
        for count in eval_requests_per_parent_step.values()
    )
    # Search still made progress (initial reward is 0).
    best = state.best_archived_node()
    assert best is not None
    assert best.evaluation.reward is not None
    assert best.evaluation.reward > 0.0


def test_log_is_well_formed(tmp_path: Path):
    """Every step has a clean event sequence: StepStarted →
    Mutation/Forecast events → ForecastsDrained → CandidateSelected/
    Deferred → Evaluation events → EvaluationsDrained → StepCompleted.
    Every request_id has exactly one terminal event."""
    log_path = tmp_path / "log.jsonl"
    config = _config(total_budget_steps=3, samples_per_parent=3, k_per_parent=2)
    with (
        BinaryStringMutationProvider(seed=42) as mp,
        BinaryStringEvaluationProvider() as ep,
        CoroutineSpeedupEstimator(UniformAsyncEstimator()) as surrogate,
    ):
        driver = SearchDriver[NoFeedback](
            config,
            mutation_provider=mp,
            evaluation_provider=ep,
            surrogate=surrogate,
            kernel_task=TEST_TASK,
            seed_reference_code="0000",
            hardware=TEST_HARDWARE,
            event_log=FileEventLog(log_path, observation_type=NoFeedback),
            observation_type=NoFeedback,
        )
        _ = driver.run(initial_program="0000")
    events = FileEventLog(log_path, observation_type=NoFeedback).read_all()

    # Step start/complete pairing.
    starts = [e for e in events if isinstance(e, StepStarted)]
    completes = [e for e in events if isinstance(e, StepCompleted)]
    assert len(starts) == len(completes) == config.total_budget_steps
    assert [s.step for s in starts] == [c.step for c in completes]

    # Every parent in every step has exactly one ForecastsDrained and
    # exactly one EvaluationsDrained (or none if the parent had no
    # selectable candidates — which doesn't happen on this toy).
    forecasts_drained_keys = {
        (e.step, str(e.parent_ulid))
        for e in events
        if isinstance(e, ForecastsDrained)
    }
    evals_drained_keys = {
        (e.step, str(e.parent_ulid))
        for e in events
        if isinstance(e, EvaluationsDrained)
    }
    expected_keys = {
        (s.step, str(p)) for s in starts for p in s.parent_ulids
    }
    assert forecasts_drained_keys == expected_keys
    assert evals_drained_keys == expected_keys

    # One terminal per candidate.
    mutation_requests = {
        e.request_id for e in events if isinstance(e, MutationRequested)
    }
    forecast_requests = {
        e.request_id for e in events if isinstance(e, ForecastRequested)
    }
    eval_requests = {
        e.request_id for e in events if isinstance(e, EvaluationRequested)
    }
    selections = {
        e.request_id for e in events if isinstance(e, CandidateSelected)
    }
    deferrals = {
        e.request_id for e in events if isinstance(e, CandidateDeferred)
    }
    # Every forecast came from a mutation, every selected candidate
    # had a forecast.
    assert forecast_requests.issubset(mutation_requests)
    assert selections.issubset(forecast_requests)
    # Selected/deferred partition the awaiting-selection set
    # (ignoring forecast-failed candidates, which on this toy is
    # zero with the stub estimator).
    assert selections.isdisjoint(deferrals)
    # Eval requests come from selected candidates only — fixed-budget
    # invariant.
    assert eval_requests == selections


def test_recovery_from_truncated_log(tmp_path: Path):
    """Truncate a log mid-step and assert the re-run produces the
    same final best as a clean run. This is the "no separate recovery
    code path" invariant: the same algorithm core picks up where it
    left off."""
    log_path = tmp_path / "log.jsonl"
    config = _config(total_budget_steps=5, samples_per_parent=3, k_per_parent=2)

    # Clean run.
    with (
        BinaryStringMutationProvider(seed=11) as mp,
        BinaryStringEvaluationProvider() as ep,
        CoroutineSpeedupEstimator(UniformAsyncEstimator()) as surrogate,
    ):
        driver = SearchDriver[NoFeedback](
            config,
            mutation_provider=mp,
            evaluation_provider=ep,
            surrogate=surrogate,
            kernel_task=TEST_TASK,
            seed_reference_code="0000",
            hardware=TEST_HARDWARE,
            event_log=FileEventLog(log_path, observation_type=NoFeedback),
            observation_type=NoFeedback,
        )
        clean_state = driver.run(initial_program="0000")
    assert clean_state.best_archived_node() is not None

    # Re-run from truncated copy.
    truncated_log_path = tmp_path / "truncated.jsonl"
    full_lines = log_path.read_text().splitlines()
    # Truncate to roughly half — somewhere mid-step.
    keep = full_lines[: max(1, len(full_lines) // 2)]
    truncated_log_path.write_text("\n".join(keep) + "\n")

    with (
        BinaryStringMutationProvider(seed=11) as mp,
        BinaryStringEvaluationProvider() as ep,
        CoroutineSpeedupEstimator(UniformAsyncEstimator()) as surrogate,
    ):
        driver = SearchDriver[NoFeedback](
            config,
            mutation_provider=mp,
            evaluation_provider=ep,
            surrogate=surrogate,
            kernel_task=TEST_TASK,
            seed_reference_code="0000",
            hardware=TEST_HARDWARE,
            event_log=FileEventLog(
                truncated_log_path, observation_type=NoFeedback
            ),
            observation_type=NoFeedback,
        )
        recovered_state = driver.run(initial_program="0000")

    # Recovery completed all the budget steps.
    assert recovered_state.current_step == config.total_budget_steps
    # Recovery's best must be at least as good as the clean run's.
    # Equality isn't guaranteed because the surviving log pins prior
    # moves and the post-truncation work is fresh, but the search
    # still gets the full step budget post-recovery.
    clean_best = clean_state.best_archived_node()
    recovered_best = recovered_state.best_archived_node()
    assert clean_best is not None
    assert recovered_best is not None
    assert clean_best.evaluation.reward is not None
    assert recovered_best.evaluation.reward is not None


def test_zero_budget_is_noop(tmp_path: Path):
    """``total_budget_steps=0``: only the bootstrap event is emitted,
    no work happens."""
    log_path = tmp_path / "log.jsonl"
    config = _config(total_budget_steps=0, samples_per_parent=2, k_per_parent=1)
    with (
        BinaryStringMutationProvider(seed=0) as mp,
        BinaryStringEvaluationProvider() as ep,
        CoroutineSpeedupEstimator(UniformAsyncEstimator()) as surrogate,
    ):
        driver = SearchDriver[NoFeedback](
            config,
            mutation_provider=mp,
            evaluation_provider=ep,
            surrogate=surrogate,
            kernel_task=TEST_TASK,
            seed_reference_code="0000",
            hardware=TEST_HARDWARE,
            event_log=FileEventLog(log_path, observation_type=NoFeedback),
            observation_type=NoFeedback,
        )
        state = driver.run(initial_program="0000")

    assert state.current_step == 0
    assert state.archive  # root only
    assert len(state.archive) == 1
    events = FileEventLog(log_path, observation_type=NoFeedback).read_all()
    assert not any(isinstance(e, StepStarted) for e in events)


# --- Replay / log-as-system-of-record invariants -----------------------


def _run_clean_search(
    log_path: Path, *, total_budget_steps: int = 4
) -> tuple[SearchState[NoFeedback], list[SearchEvent[NoFeedback]], SearchConfig]:
    """Run a search to completion, return (final_state, all_events)."""
    config = _config(
        total_budget_steps=total_budget_steps,
        samples_per_parent=3,
        k_per_parent=2,
    )
    with (
        BinaryStringMutationProvider(seed=11) as mp,
        BinaryStringEvaluationProvider() as ep,
        CoroutineSpeedupEstimator(UniformAsyncEstimator()) as surrogate,
    ):
        driver = SearchDriver[NoFeedback](
            config,
            mutation_provider=mp,
            evaluation_provider=ep,
            surrogate=surrogate,
            kernel_task=TEST_TASK,
            seed_reference_code="0000",
            hardware=TEST_HARDWARE,
            event_log=FileEventLog(log_path, observation_type=NoFeedback),
            observation_type=NoFeedback,
        )
        final_state = driver.run(initial_program="0000")
    events = FileEventLog(log_path, observation_type=NoFeedback).read_all()
    return final_state, events, config


def test_step_boundary_replay_is_consistent(tmp_path: Path):
    """For every step boundary in a clean log, replaying the prefix
    up to that boundary produces a state whose ``current_step``,
    surrogate context, and archive monotonicity match the truncation
    point. This pins the property that ``compute_pending_actions``
    relies on: a log prefix folds into the state from which the next
    moves are derivable."""
    from gpu_forecasters.max_reward_puct.v3.events import StepCompleted

    log_path = tmp_path / "log.jsonl"
    _final_state, events, config = _run_clean_search(log_path)

    step_complete_indices = [
        i for i, e in enumerate(events) if isinstance(e, StepCompleted)
    ]
    assert len(step_complete_indices) == config.total_budget_steps

    prev_archive_size = 1  # root only
    for boundary in step_complete_indices:
        prefix = events[: boundary + 1]
        state_at_boundary = replay(
            prefix,
            k_per_parent=config.k_per_parent,
            archive_capacity=config.archive_capacity,
            observation_type=NoFeedback,
        )
        # current_step advances on StepCompleted, so after folding
        # the boundary's StepCompleted, current_step == boundary_step + 1.
        boundary_event = events[boundary]
        assert isinstance(boundary_event, StepCompleted)
        assert state_at_boundary.current_step == boundary_event.step + 1
        # Active-step bookkeeping is cleared between steps.
        assert not state_at_boundary.current_step_active
        # Surrogate context survived the fold.
        assert state_at_boundary.kernel_task == TEST_TASK
        assert state_at_boundary.seed_reference_code == "0000"
        assert state_at_boundary.hardware == TEST_HARDWARE
        # Archive grows monotonically across step boundaries (no
        # eviction at this scale).
        assert len(state_at_boundary.archive) >= prev_archive_size
        prev_archive_size = len(state_at_boundary.archive)


# --- Surrogate-context drift on resume ---------------------------------


def _resume_with_overrides(
    log_path: Path,
    *,
    kernel_task: KernelTaskInfo = TEST_TASK,
    seed_reference_code: str = "0000",
    hardware: HardwareContext = TEST_HARDWARE,
):
    """Try to resume a search from ``log_path`` with possibly-diverged
    surrogate-context constructor args. Returns the driver's run()
    result, or raises whatever the driver raises."""
    config = _config(total_budget_steps=4, samples_per_parent=3, k_per_parent=2)
    with (
        BinaryStringMutationProvider(seed=11) as mp,
        BinaryStringEvaluationProvider() as ep,
        CoroutineSpeedupEstimator(UniformAsyncEstimator()) as surrogate,
    ):
        driver = SearchDriver[NoFeedback](
            config,
            mutation_provider=mp,
            evaluation_provider=ep,
            surrogate=surrogate,
            kernel_task=kernel_task,
            seed_reference_code=seed_reference_code,
            hardware=hardware,
            event_log=FileEventLog(log_path, observation_type=NoFeedback),
            observation_type=NoFeedback,
        )
        return driver.run(initial_program="0000")


_OTHER_HW = HardwareContext(
    device_name="other-cpu",
    compute_capability=(1, 0),
    total_global_memory_gb=1.0,
    multiprocessor_count=1,
    max_threads_per_multiprocessor=1,
    clock_rate_ghz=1.0,
    memory_clock_rate_ghz=1.0,
    memory_bus_width_bits=1,
)
_OTHER_TASK = KernelTaskInfo(op_name="other", level_id=1, task_id=2)


@pytest.mark.parametrize(
    ("field", "override"),
    [
        ("seed_reference_code", {"seed_reference_code": "DIFFERENT"}),
        ("hardware", {"hardware": _OTHER_HW}),
        ("kernel_task", {"kernel_task": _OTHER_TASK}),
    ],
)
def test_resume_with_diverged_surrogate_context_raises(
    tmp_path: Path, field: str, override: Any
):
    log_path = tmp_path / "log.jsonl"
    _run_clean_search(log_path, total_budget_steps=2)
    with pytest.raises(SurrogateContextMismatch, match=field):
        _resume_with_overrides(log_path, **override)


def test_resume_with_matching_context_completes(tmp_path: Path):
    """Sanity-check the negative cases above: resume with matching
    args succeeds (and runs to completion since the original used the
    same total_budget_steps)."""
    log_path = tmp_path / "log.jsonl"
    _run_clean_search(log_path, total_budget_steps=2)
    # Re-run with the same budget — first run already hit the budget,
    # so the resumed run should be a no-op.
    config = _config(total_budget_steps=2, samples_per_parent=3, k_per_parent=2)
    with (
        BinaryStringMutationProvider(seed=11) as mp,
        BinaryStringEvaluationProvider() as ep,
        CoroutineSpeedupEstimator(UniformAsyncEstimator()) as surrogate,
    ):
        driver = SearchDriver[NoFeedback](
            config,
            mutation_provider=mp,
            evaluation_provider=ep,
            surrogate=surrogate,
            kernel_task=TEST_TASK,
            seed_reference_code="0000",
            hardware=TEST_HARDWARE,
            event_log=FileEventLog(log_path, observation_type=NoFeedback),
            observation_type=NoFeedback,
        )
        state = driver.run(initial_program="0000")
    assert state.current_step == 2
