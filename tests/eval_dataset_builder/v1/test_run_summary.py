"""Tests for ``compute_run_summary_from_event_log``."""

from __future__ import annotations

from ulid import ULID

from gpu_forecasters.eval_dataset_builder.v1.summary import (
    compute_run_summary_from_event_log,
)
from gpu_forecasters.gpu_mode_kernel.core import (
    CompileFailedFeedback,
    GpuModeKernelObservation,
    InfrastructureFailureFeedback,
    SuccessFeedback,
)
from gpu_forecasters.gpu_mode_kernel.packs.trimul import TriMulCaseSpeedup
from gpu_forecasters.hill_climbing.domain import Evaluation, Node
from gpu_forecasters.landscape_map.v1.domain import SpeedupBin
from gpu_forecasters.max_reward_puct.v2.events import (
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


_OBS = GpuModeKernelObservation[TriMulCaseSpeedup]


def _success(speedup: float) -> Evaluation[GpuModeKernelObservation[TriMulCaseSpeedup]]:
    feedback: SuccessFeedback[TriMulCaseSpeedup] = SuccessFeedback(
        aggregated_speedup=speedup,
        aggregation_method="geomean",
        per_case_speedups=[],
    )
    return Evaluation[GpuModeKernelObservation[TriMulCaseSpeedup]](
        observation=_OBS(feedback=feedback),
        reward=-abs(speedup - 3.0),  # Goal-conditioned-style score; sign matters.
    )


def _compile_fail() -> Evaluation[GpuModeKernelObservation[TriMulCaseSpeedup]]:
    return Evaluation[GpuModeKernelObservation[TriMulCaseSpeedup]](
        observation=_OBS(feedback=CompileFailedFeedback(compilation_error="boom")),
        reward=None,
    )


def _infra_fail() -> Evaluation[GpuModeKernelObservation[TriMulCaseSpeedup]]:
    return Evaluation[GpuModeKernelObservation[TriMulCaseSpeedup]](
        observation=_OBS(feedback=InfrastructureFailureFeedback(reason="modal")),
        reward=None,
    )


def test_counts_per_bin_and_archive() -> None:
    target_bin = SpeedupBin.HIGH_SPEEDUP  # 2.83×–4.00×, midpoint ≈ 3.36
    seed_eval = _success(3.0)  # in target bin
    seed_node = Node[GpuModeKernelObservation[TriMulCaseSpeedup]](
        program_code="SEED",
        ancestors=[],
        evaluation=seed_eval,
        is_seed=True,
    )

    in_target_eval = _success(3.2)  # HIGH_SPEEDUP
    out_target_eval = _success(8.0)  # EXTREME_SPEEDUP

    parent_ulid = seed_node.ulid
    child_in_id = str(ULID())
    child_out_id = str(ULID())
    child_compile_fail_id = str(ULID())
    eval_failed_id = str(ULID())

    events: list[SearchEvent[GpuModeKernelObservation[TriMulCaseSpeedup]]] = [
        SearchInitialized[GpuModeKernelObservation[TriMulCaseSpeedup]](root=seed_node),
        StepStarted(step=0, parent_ulids=[parent_ulid]),
        MutationRequested(request_id="m1", parent_ulid=parent_ulid),
        MutationRequested(request_id="m2", parent_ulid=parent_ulid),
        MutationRequested(request_id="m3", parent_ulid=parent_ulid),
        MutationRequested(request_id="m4", parent_ulid=parent_ulid),
        MutationCompleted(request_id="m1", code="C1"),
        MutationCompleted(request_id="m2", code="C2"),
        MutationCompleted(request_id="m3", code="C3"),
        MutationFailed(request_id="m4", reason="LLM bust"),
        EvaluationRequested(
            request_id=child_in_id,
            child_ulid=ULID(),
            parent_ulid=parent_ulid,
            code="C1",
            from_mutation_request_id="m1",
        ),
        EvaluationRequested(
            request_id=child_out_id,
            child_ulid=ULID(),
            parent_ulid=parent_ulid,
            code="C2",
            from_mutation_request_id="m2",
        ),
        EvaluationRequested(
            request_id=child_compile_fail_id,
            child_ulid=ULID(),
            parent_ulid=parent_ulid,
            code="C3",
            from_mutation_request_id="m3",
        ),
        EvaluationCompleted[GpuModeKernelObservation[TriMulCaseSpeedup]](
            request_id=child_in_id, evaluation=in_target_eval
        ),
        EvaluationCompleted[GpuModeKernelObservation[TriMulCaseSpeedup]](
            request_id=child_out_id, evaluation=out_target_eval
        ),
        EvaluationCompleted[GpuModeKernelObservation[TriMulCaseSpeedup]](
            request_id=child_compile_fail_id, evaluation=_compile_fail()
        ),
        EvaluationFailed(request_id=eval_failed_id, reason="modal timeout"),
        StepCompleted(step=0),
    ]

    summary = compute_run_summary_from_event_log(
        events,
        target_bin=target_bin,
        target_band_lo=2.83,
        target_band_hi=4.0,
        target_midpoint_speedup=3.36,
        seed_source_id="seed/abc",
        seed_speedup_at_harvest=3.0,
        model_slug="fake-model",
        search_config={"k_per_parent": 4},
        k_per_parent=4,
        archive_capacity=32,
        wall_clock_seconds=1.5,
        observation_type=_OBS,
    )

    assert summary.total_candidates_evaluated == 3  # only EvaluationCompleted counts
    assert summary.per_bin_count_all_candidates.get("HIGH_SPEEDUP") == 1
    assert summary.per_bin_count_all_candidates.get("EXTREME_SPEEDUP") == 1
    assert summary.per_bin_count_all_candidates.get("FAILURE") == 1
    assert summary.in_target_bin_count_all_candidates == 1

    # Archive: seed (HIGH_SPEEDUP) + 1 in-target child + 1 out-of-target child
    # under generous capacity. Compile-failed child has reward None and is
    # dropped by the archive update.
    assert summary.in_target_bin_count_archive_at_end >= 1
    archive_total = sum(summary.per_bin_count_archive_at_end.values())
    assert archive_total >= 2

    assert summary.seed_speedup_after_bootstrap_eval == 3.0


def test_infrastructure_failure_classified_separately() -> None:
    target_bin = SpeedupBin.HIGH_SPEEDUP
    seed_node = Node[GpuModeKernelObservation[TriMulCaseSpeedup]](
        program_code="SEED",
        ancestors=[],
        evaluation=_success(3.0),
        is_seed=True,
    )
    parent_ulid = seed_node.ulid
    events: list[SearchEvent[GpuModeKernelObservation[TriMulCaseSpeedup]]] = [
        SearchInitialized[GpuModeKernelObservation[TriMulCaseSpeedup]](root=seed_node),
        StepStarted(step=0, parent_ulids=[parent_ulid]),
        MutationRequested(request_id="m1", parent_ulid=parent_ulid),
        MutationCompleted(request_id="m1", code="C1"),
        EvaluationRequested(
            request_id="e1",
            child_ulid=ULID(),
            parent_ulid=parent_ulid,
            code="C1",
            from_mutation_request_id="m1",
        ),
        EvaluationCompleted[GpuModeKernelObservation[TriMulCaseSpeedup]](
            request_id="e1", evaluation=_infra_fail()
        ),
        StepCompleted(step=0),
    ]
    summary = compute_run_summary_from_event_log(
        events,
        target_bin=target_bin,
        target_band_lo=2.83,
        target_band_hi=4.0,
        target_midpoint_speedup=3.36,
        seed_source_id="seed/abc",
        seed_speedup_at_harvest=3.0,
        model_slug="fake-model",
        search_config={},
        k_per_parent=4,
        archive_capacity=32,
        wall_clock_seconds=0.0,
        observation_type=_OBS,
    )
    assert summary.per_bin_count_all_candidates.get("INFRASTRUCTURE_FAILURE") == 1
    assert summary.per_bin_count_all_candidates.get("FAILURE") is None
