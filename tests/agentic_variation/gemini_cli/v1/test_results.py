"""Unit tests for result loaders and summary aggregation."""

from __future__ import annotations

from pathlib import Path

from gpu_forecasters.agentic_variation.gemini_cli.v1.models import (
    TrajectoryRecord,
    TrimulRunResult,
)
from gpu_forecasters.agentic_variation.gemini_cli.v1.results import (
    RESULT_FILENAME,
    best_record,
    compute_summary,
)
from gpu_forecasters.trimul.core import (
    CaseSpeedup,
    CompileFailedFeedback,
    SuccessFeedback,
    TriMulKernelExecutionFeedback,
)


def _trajectory_record(
    feedback: TriMulKernelExecutionFeedback, sha: str = "abc"
) -> TrajectoryRecord:
    return TrajectoryRecord(
        timestamp_utc="2026-04-18T00:00:00+00:00",
        path="kernel_v1.py",
        sha256=sha,
        kernel_source="def custom_kernel(data): ...",
        feedback=feedback,
    )


def _success(speedup: float) -> SuccessFeedback:
    return SuccessFeedback(
        aggregated_speedup=speedup,
        aggregation_method="geomean",
        per_case_speedups=[
            CaseSpeedup(
                seqlen=256,
                bs=1,
                dim=128,
                hiddendim=128,
                nomask=True,
                distribution="normal",
                speedup=speedup,
                runtime_ns=1_000.0,
                ref_runtime_ns=1_000.0 * speedup,
            )
        ],
    )


def test_best_record_empty() -> None:
    assert best_record([]) is None


def test_best_record_all_failures() -> None:
    records = [
        _trajectory_record(CompileFailedFeedback(compilation_error="x")),
        _trajectory_record(CompileFailedFeedback(compilation_error="y")),
    ]
    assert best_record(records) is None


def test_best_record_picks_highest_speedup() -> None:
    records = [
        _trajectory_record(_success(0.8), sha="slow"),
        _trajectory_record(_success(2.5), sha="fast"),
        _trajectory_record(CompileFailedFeedback(compilation_error="x"), sha="bad"),
        _trajectory_record(_success(1.4), sha="mid"),
    ]
    best = best_record(records)
    assert best is not None
    assert best.sha256 == "fast"


def _write_result(
    run_dir: Path, best_speedup: float | None, best_kernel_sha256: str | None
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    result = TrimulRunResult(
        exit_code=0,
        elapsed_s=1.0,
        final_kernel_source=None,
        best_speedup=best_speedup,
        best_kernel_sha256=best_kernel_sha256,
    )
    _ = (run_dir / RESULT_FILENAME).write_text(result.model_dump_json())


def test_compute_summary_zero_runs(tmp_path: Path) -> None:
    summary = compute_summary(tmp_path, expected_num_runs=3)
    assert summary.expected_num_runs == 3
    assert summary.completed_num_runs == 0
    assert summary.num_with_success == 0
    assert summary.best_speedup_per_run == []
    assert summary.min is None
    assert summary.median is None
    assert summary.max is None


def test_compute_summary_mixed_success_and_failure(tmp_path: Path) -> None:
    # run_00: success at 1.5x; run_01: no successful candidate;
    # run_02: success at 2.5x. Partial run_03 with no result.json should
    # be ignored.
    _write_result(tmp_path / "run_00", best_speedup=1.5, best_kernel_sha256="a")
    _write_result(tmp_path / "run_01", best_speedup=None, best_kernel_sha256=None)
    _write_result(tmp_path / "run_02", best_speedup=2.5, best_kernel_sha256="c")
    (tmp_path / "run_03").mkdir()  # partial: no result.json
    _ = (tmp_path / "run_03" / "trajectory.jsonl").write_text("")

    summary = compute_summary(tmp_path, expected_num_runs=4)
    assert summary.expected_num_runs == 4
    assert summary.completed_num_runs == 3  # run_03 excluded
    assert summary.num_with_success == 2
    assert summary.best_speedup_per_run == [1.5, None, 2.5]
    assert summary.min == 1.5
    assert summary.max == 2.5
    assert summary.median == 2.0
