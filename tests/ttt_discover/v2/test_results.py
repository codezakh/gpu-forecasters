from pathlib import Path

from gpu_forecasters.trimul.core import (
    CaseSpeedup,
    IncorrectFeedback,
    InfrastructureFailureFeedback,
    SuccessFeedback,
)
from gpu_forecasters.ttt_discover.v2.domain.candidate import CandidateId
from gpu_forecasters.ttt_discover.v2.domain.outcome import (
    ParseFailureFeedback,
    TriMulRLOutcome,
)
from gpu_forecasters.ttt_discover.v2.domain.records import RolloutRecord
from gpu_forecasters.ttt_discover.v2.results import V2ExperimentResults
from gpu_forecasters.ttt_discover.v2.sinks.jsonl import JsonlRolloutSink


def _make_record(
    step: int,
    rollout_index: int,
    outcome: TriMulRLOutcome,
    reward: float,
    candidate_id: str,
) -> RolloutRecord:
    return RolloutRecord(
        step=step,
        group_index=0,
        rollout_index=rollout_index,
        timestamp_utc="2026-04-24T00:00:00+00:00",
        parent_id=None,
        candidate_id=CandidateId(candidate_id),
        task_prompt="t",
        feedback_prompt="",
        raw_response="r",
        parsed_code=None,
        outcome=outcome,
        reward=reward,
        prompt_tokens=0,
        response_tokens=0,
        sampling_time_s=0.0,
        eval_time_s=0.0,
    )


def _make_success(runtime_ns: float) -> SuccessFeedback:
    return SuccessFeedback(
        aggregated_speedup=1.0,
        aggregation_method="geomean",
        per_case_speedups=[
            CaseSpeedup(
                seqlen=256,
                bs=2,
                dim=128,
                hiddendim=128,
                nomask=True,
                distribution="normal",
                speedup=1.0,
                runtime_ns=runtime_ns,
                ref_runtime_ns=runtime_ns,
            )
        ],
    )


def _write_seed_rollouts(
    output_dir: Path, seed_index: int, records: list[RolloutRecord]
) -> None:
    seed_dir = output_dir / "tinker_log" / f"seed_{seed_index:02d}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    sink = JsonlRolloutSink(path=seed_dir / "rollouts.jsonl")
    for r in records:
        sink.record(r)
    sink.close()


def test_summary_counts_by_kind(tmp_path: Path) -> None:
    records = [
        _make_record(0, 0, _make_success(2_500_000.0), 1.0, "c0"),
        _make_record(0, 1, ParseFailureFeedback(reason="x"), 0.0, "c1"),
        _make_record(0, 2, IncorrectFeedback(error_message="x"), 0.0, "c2"),
        _make_record(1, 0, InfrastructureFailureFeedback(reason="x"), 0.0, "c3"),
    ]
    _write_seed_rollouts(tmp_path, 0, records)

    results = V2ExperimentResults(tmp_path)
    seeds = results.per_seed()
    assert len(seeds) == 1
    summary = seeds[0].summary()
    assert summary.num_rollouts == 4
    assert summary.num_successes == 1
    assert summary.num_parse_failures == 1
    assert summary.num_incorrect == 1
    assert summary.num_infra_failures == 1
    assert summary.best_reward == 1.0


def test_best_by_step_is_monotone(tmp_path: Path) -> None:
    records = [
        _make_record(0, 0, _make_success(5_000_000.0), 0.5, "c0"),  # best so far = 0.5
        _make_record(0, 1, ParseFailureFeedback(reason="x"), 0.0, "c1"),
        _make_record(1, 0, _make_success(2_500_000.0), 1.0, "c2"),  # best so far = 1.0
        _make_record(2, 0, IncorrectFeedback(error_message="x"), 0.0, "c3"),
        _make_record(2, 1, _make_success(10_000_000.0), 0.25, "c4"),
        _make_record(3, 0, _make_success(1_250_000.0), 2.0, "c5"),
    ]
    _write_seed_rollouts(tmp_path, 0, records)
    seeds = V2ExperimentResults(tmp_path).per_seed()
    bests = seeds[0].best_by_step()
    steps = [b.step for b in bests]
    rewards = [b.best_reward for b in bests]
    assert steps == [0, 1, 2, 3]
    # Monotone non-decreasing.
    assert rewards == sorted(rewards)
    assert rewards == [0.5, 1.0, 1.0, 2.0]


def test_successful_rollouts_filter(tmp_path: Path) -> None:
    records = [
        _make_record(0, 0, _make_success(2_500_000.0), 1.0, "c0"),
        _make_record(0, 1, ParseFailureFeedback(reason="x"), 0.0, "c1"),
    ]
    _write_seed_rollouts(tmp_path, 0, records)
    seeds = V2ExperimentResults(tmp_path).per_seed()
    assert len(seeds[0].successful_rollouts()) == 1
    assert seeds[0].successful_rollouts()[0].candidate_id == "c0"


def test_no_seeds_when_tinker_log_absent(tmp_path: Path) -> None:
    assert V2ExperimentResults(tmp_path).per_seed() == []
