"""Read-side view of a run's on-disk artifacts.

Kept deliberately free of the heavy runtime deps (``docker``,
``fastmcp``) that :mod:`orchestrator` pulls in — analysis code and
comparison scripts should be able to import this module on a laptop
without a container runtime. ``orchestrator`` itself imports
:func:`load_trajectory` / :func:`best_record` from here at run end.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from pathlib import Path

from gpu_forecasters.trimul.core import SuccessFeedback

from .models import RepeatedRunSummary, TrajectoryRecord, TrimulRunResult


RESULT_FILENAME = "result.json"
TRAJECTORY_FILENAME = "trajectory.jsonl"
RUN_DIR_PREFIX = "run_"


def load_trajectory(run_dir: Path) -> list[TrajectoryRecord]:
    """Parse the per-call scoring log written by ``trimul_score_server``."""
    path = run_dir / TRAJECTORY_FILENAME
    if not path.is_file():
        return []
    records: list[TrajectoryRecord] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(TrajectoryRecord.model_validate_json(line))
    return records


def best_record(records: list[TrajectoryRecord]) -> TrajectoryRecord | None:
    """Highest-aggregated-speedup ``SuccessFeedback`` record, or ``None``."""
    best: TrajectoryRecord | None = None
    best_speedup = float("-inf")
    for r in records:
        if isinstance(r.feedback, SuccessFeedback):
            if r.feedback.aggregated_speedup > best_speedup:
                best_speedup = r.feedback.aggregated_speedup
                best = r
    return best


@dataclass(frozen=True)
class RunArtifacts:
    """One ``run_<id>/`` directory's contents, eagerly loaded.

    Read-side view — never serialized back to disk. ``run_dir`` is
    exposed so callers can reach for side artifacts (``agent_raw.log``,
    ``kernel_v*.py``, rendered prompts) without re-deriving the path.
    """

    run_dir: Path
    result: TrimulRunResult
    trajectory: list[TrajectoryRecord]

    @classmethod
    def load(cls, run_dir: Path) -> RunArtifacts:
        result = TrimulRunResult.model_validate_json(
            (run_dir / RESULT_FILENAME).read_text()
        )
        return cls(
            run_dir=run_dir,
            result=result,
            trajectory=load_trajectory(run_dir),
        )


def _iter_run_dirs(output_dir: Path) -> list[Path]:
    """Run dirs in chronological / index order (both naming schemes sort naturally)."""
    return sorted(p for p in output_dir.glob(f"{RUN_DIR_PREFIX}*") if p.is_dir())


def load_run_artifacts(output_dir: Path) -> list[RunArtifacts]:
    """Load every completed ``run_*/result.json`` under ``output_dir``."""
    return [
        RunArtifacts.load(d)
        for d in _iter_run_dirs(output_dir)
        if (d / RESULT_FILENAME).is_file()
    ]


def compute_summary(
    output_dir: Path, expected_num_runs: int
) -> RepeatedRunSummary:
    """Aggregate completed ``run_*/result.json`` files into a summary.

    Derived from disk, not from any in-memory accumulator — so a
    partially-resumed sweep produces the same summary as a clean run
    once both have the same set of completed runs on disk.
    """
    artifacts = load_run_artifacts(output_dir)
    best_per_run = [a.result.best_speedup for a in artifacts]
    speedups = [s for s in best_per_run if s is not None]
    return RepeatedRunSummary(
        expected_num_runs=expected_num_runs,
        completed_num_runs=len(artifacts),
        num_with_success=len(speedups),
        best_speedup_per_run=best_per_run,
        min=min(speedups) if speedups else None,
        median=statistics.median(speedups) if speedups else None,
        max=max(speedups) if speedups else None,
    )
