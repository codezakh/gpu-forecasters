"""End-to-end v2 search tests against the binary-string toy problem.

Same toy as ``tests/max_reward_puct/test_search.py``. Providers are
written *natively* against the v2 async protocols — per-candidate
``submit`` returning ``Future``. No wrapping of sync providers.
"""

from __future__ import annotations

import random
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Self

from arid_badger.hill_climbing.domain import Evaluation, NoFeedback
from arid_badger.max_reward_puct.v2.config import SearchConfig
from arid_badger.max_reward_puct.v2.event_log import FileEventLog
from arid_badger.max_reward_puct.v2.events import (
    EvaluationCompleted,
    EvaluationFailed,
    EvaluationRequested,
    MutationCompleted,
    MutationFailed,
    MutationRequested,
    StepCompleted,
    StepStarted,
)
from arid_badger.max_reward_puct.v2.search import SearchDriver


def _eval(reward: float | None) -> Evaluation[NoFeedback]:
    return Evaluation(observation=NoFeedback(), reward=reward)


# ---------------------------------------------------------------------------
# Native async providers for the binary-string toy.
# ---------------------------------------------------------------------------


class BinaryStringMutationProvider:
    """One submit → one candidate. Picks a random bit, flips it.

    Thread-safe: the RNG is guarded by a lock since submit may be
    called concurrently in tests that exercise overlap.
    """

    def __init__(self, seed: int | None = None, max_workers: int = 8) -> None:
        self._rng = random.Random(seed)
        self._rng_lock = threading.Lock()
        self._max_workers = max_workers
        self._executor: ThreadPoolExecutor | None = None

    def submit(
        self,
        parent_code: str,
        evaluation: Evaluation[NoFeedback],
    ) -> Future[str]:
        assert self._executor is not None, (
            "BinaryStringMutationProvider must be entered before submit"
        )
        return self._executor.submit(self._mutate, parent_code)

    def _mutate(self, parent_code: str) -> str:
        with self._rng_lock:
            pos = self._rng.randrange(len(parent_code))
        bits = list(parent_code)
        bits[pos] = "1" if bits[pos] == "0" else "0"
        return "".join(bits)

    def __enter__(self) -> Self:
        self._executor = ThreadPoolExecutor(max_workers=self._max_workers)
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None


class BinaryStringEvaluationProvider:
    """One submit → one evaluation. Reward = int(code, 2)."""

    def __init__(
        self, max_workers: int = 8, sleep_s: float = 0.0
    ) -> None:
        self._max_workers = max_workers
        self._sleep_s = sleep_s
        self._executor: ThreadPoolExecutor | None = None
        self.timeline: list[tuple[str, float, float]] = []
        self._timeline_lock = threading.Lock()

    def submit(self, program_code: str) -> Future[Evaluation[NoFeedback]]:
        assert self._executor is not None, (
            "BinaryStringEvaluationProvider must be entered before submit"
        )
        return self._executor.submit(self._evaluate, program_code)

    def _evaluate(self, program_code: str) -> Evaluation[NoFeedback]:
        started = time.perf_counter()
        if self._sleep_s > 0:
            time.sleep(self._sleep_s)
        ended = time.perf_counter()
        with self._timeline_lock:
            self.timeline.append((program_code, started, ended))
        try:
            reward: float | None = float(int(program_code, 2))
        except ValueError:
            reward = None
        return _eval(reward)

    def __enter__(self) -> Self:
        self._executor = ThreadPoolExecutor(max_workers=self._max_workers)
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _run(
    *,
    log_path: Path,
    total_budget_steps: int,
    samples_per_parent: int = 4,
    batch_size: int = 1,
    k_per_parent: int = 2,
    seed: int = 42,
    eval_sleep_s: float = 0.0,
    initial_program: str = "0000",
):
    config = SearchConfig(
        total_budget_steps=total_budget_steps,
        batch_size=batch_size,
        samples_per_parent=samples_per_parent,
        k_per_parent=k_per_parent,
    )
    with BinaryStringMutationProvider(seed=seed) as mp, BinaryStringEvaluationProvider(
        sleep_s=eval_sleep_s
    ) as ep:
        driver = SearchDriver[NoFeedback](
            config,
            mutation_provider=mp,
            evaluation_provider=ep,
            event_log=FileEventLog(log_path, observation_type=NoFeedback),
            observation_type=NoFeedback,
        )
        return driver.run(initial_program=initial_program), ep


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_search_converges_to_maximum(tmp_path: Path):
    state, _ = _run(log_path=tmp_path / "log.jsonl", total_budget_steps=20)
    best = state.best_archived_node()
    assert best is not None
    assert best.program_code == "1111"
    assert best.evaluation.reward == 15.0


def test_batch_size_greater_than_one(tmp_path: Path):
    state, _ = _run(
        log_path=tmp_path / "log.jsonl", total_budget_steps=20, batch_size=2
    )
    best = state.best_archived_node()
    assert best is not None
    assert best.evaluation.reward is not None
    assert best.evaluation.reward >= 10.0


def test_log_is_well_formed(tmp_path: Path):
    """Every StepStarted has a matching StepCompleted; every request has a
    terminal. A clean run produces a clean log."""
    log_path = tmp_path / "log.jsonl"
    _ = _run(log_path=log_path, total_budget_steps=5)

    events = FileEventLog(log_path, observation_type=NoFeedback).read_all()

    starts = [e for e in events if isinstance(e, StepStarted)]
    completes = [e for e in events if isinstance(e, StepCompleted)]
    assert len(starts) == len(completes)
    assert [s.step for s in starts] == [c.step for c in completes]

    mutation_requests = {
        e.request_id for e in events if isinstance(e, MutationRequested)
    }
    mutation_completes = {
        e.request_id for e in events if isinstance(e, MutationCompleted)
    }
    assert mutation_completes == mutation_requests

    eval_requests = {
        e.request_id for e in events if isinstance(e, EvaluationRequested)
    }
    eval_completes = {
        e.request_id for e in events if isinstance(e, EvaluationCompleted)
    }
    assert eval_completes == eval_requests


def test_resume_from_complete_log_is_noop(tmp_path: Path):
    """Running the driver twice with the same budget shouldn't dispatch any
    new work on the second run."""
    log_path = tmp_path / "log.jsonl"
    _ = _run(log_path=log_path, total_budget_steps=5)
    after_first = FileEventLog(log_path, observation_type=NoFeedback).read_all()

    _ = _run(log_path=log_path, total_budget_steps=5)
    after_second = FileEventLog(log_path, observation_type=NoFeedback).read_all()

    assert len(after_first) == len(after_second)


def test_resume_from_partial_run_extends_log(tmp_path: Path):
    """Run 3 steps, then bump budget to 8. Second run picks up and
    only dispatches the remaining work."""
    log_path = tmp_path / "log.jsonl"
    _ = _run(log_path=log_path, total_budget_steps=3)
    after_first = FileEventLog(log_path, observation_type=NoFeedback).read_all()

    state, _ = _run(log_path=log_path, total_budget_steps=8)
    after_second = FileEventLog(log_path, observation_type=NoFeedback).read_all()

    assert len(after_second) > len(after_first)
    assert state.current_step == 8


def test_evaluations_run_concurrently_within_a_step(tmp_path: Path):
    """With native async providers, mutations+evaluations in the same step
    overlap in wall-clock time."""
    _, ep = _run(
        log_path=tmp_path / "log.jsonl",
        total_budget_steps=1,
        samples_per_parent=8,
        eval_sleep_s=0.05,
        initial_program="00000000",
    )
    starts = [s for _, s, _ in ep.timeline]
    ends = [e for _, _, e in ep.timeline]
    total_span = max(ends) - min(starts)
    # 8 evals @ 50ms: serial would be ~400ms, concurrent should be <200ms.
    assert total_span < 0.2, (
        f"Expected concurrent evaluation (<200ms), got {total_span:.3f}s — "
        "interleaving is broken."
    )
    # At least one pair must actually overlap in time.
    overlaps = 0
    for i in range(len(ep.timeline)):
        for j in range(i + 1, len(ep.timeline)):
            _, si, ei = ep.timeline[i]
            _, sj, ej = ep.timeline[j]
            if si < ej and sj < ei:
                overlaps += 1
    assert overlaps > 0


def test_crash_between_mutation_completed_and_evaluation_requested(tmp_path: Path):
    """The narrow window where a MutationCompleted is logged but the
    paired EvaluationRequested isn't yet. Recovery must dispatch the
    eval rather than dropping the produced code on the floor.
    """
    log_path = tmp_path / "log.jsonl"
    # Run a 1-step baseline.
    state_full, _ = _run(log_path=log_path, total_budget_steps=1)
    full_events = FileEventLog(log_path, observation_type=NoFeedback).read_all()

    # Truncate immediately after the FIRST MutationCompleted.
    first_mut_completed_idx = next(
        i for i, e in enumerate(full_events) if isinstance(e, MutationCompleted)
    )
    truncated = full_events[: first_mut_completed_idx + 1]

    # Sanity: no EvaluationRequested for that mutation in the truncated log.
    completed = full_events[first_mut_completed_idx]
    assert isinstance(completed, MutationCompleted)
    eval_reqs_for_mutation = [
        e
        for e in truncated
        if isinstance(e, EvaluationRequested)
        and e.from_mutation_request_id == completed.request_id
    ]
    assert eval_reqs_for_mutation == []

    log_path.unlink()
    truncated_log = FileEventLog(log_path, observation_type=NoFeedback)
    for e in truncated:
        truncated_log.append(e)

    # Resume.
    state_resumed, _ = _run(log_path=log_path, total_budget_steps=1)
    assert state_resumed.current_step == 1

    # Recovery must have synthesized an EvaluationRequested linked to
    # the orphaned MutationCompleted.
    resumed_events = FileEventLog(
        log_path, observation_type=NoFeedback
    ).read_all()
    linked_eval_reqs = [
        e
        for e in resumed_events
        if isinstance(e, EvaluationRequested)
        and e.from_mutation_request_id == completed.request_id
    ]
    assert len(linked_eval_reqs) == 1, (
        "Recovery should synthesize one EvaluationRequested for the "
        "stranded MutationCompleted; got "
        f"{len(linked_eval_reqs)}."
    )
    # And that eval must have a terminal.
    eval_req = linked_eval_reqs[0]
    terminal_eval_ids = {
        e.request_id
        for e in resumed_events
        if isinstance(e, EvaluationCompleted)
        or e.__class__.__name__ == "EvaluationFailed"
    }
    assert eval_req.request_id in terminal_eval_ids


def test_crash_midstep_redispatches_rather_than_dropping(tmp_path: Path):
    """Real recovery: when the log is truncated mid-step, the driver
    re-submits the un-terminated mutation/evaluation requests and
    finalizes the step with their fresh results — it does NOT emit a
    synthetic StepCompleted on incomplete state."""
    log_path = tmp_path / "log.jsonl"

    # Phase 1: run a 2-step baseline to capture a realistic log shape.
    state_full, _ = _run(log_path=log_path, total_budget_steps=2)
    full_events = FileEventLog(log_path, observation_type=NoFeedback).read_all()

    # Truncate mid-step-1: keep everything through the StepStarted for
    # step 1 plus one MutationRequested but no terminals.
    step_started_indices = [
        i for i, e in enumerate(full_events) if isinstance(e, StepStarted)
    ]
    assert len(step_started_indices) >= 2
    start_of_step_1 = step_started_indices[1]
    first_mut_req_idx = next(
        i
        for i, e in enumerate(full_events[start_of_step_1:], start=start_of_step_1)
        if isinstance(e, MutationRequested)
    )
    truncated = full_events[: first_mut_req_idx + 1]

    log_path.unlink()
    truncated_log = FileEventLog(log_path, observation_type=NoFeedback)
    for e in truncated:
        truncated_log.append(e)

    # Phase 2: resume with the same budget.
    state_resumed, _ = _run(log_path=log_path, total_budget_steps=2)

    # Resumed run must have advanced through step 1 cleanly.
    assert state_resumed.current_step == 2

    # The un-terminated mutation request's id must have been terminated
    # (either Completed or Failed) on resumption — i.e., real recovery
    # happened, not a synthetic StepCompleted that silently drops work.
    resumed_events = FileEventLog(
        log_path, observation_type=NoFeedback
    ).read_all()
    original_mut_req = full_events[first_mut_req_idx]
    assert isinstance(original_mut_req, MutationRequested)
    terminal_ids = {
        e.request_id
        for e in resumed_events
        if isinstance(e, MutationCompleted)
        or e.__class__.__name__ == "MutationFailed"
    }
    assert original_mut_req.request_id in terminal_ids
