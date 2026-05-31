"""
Per-step trajectory logging for Max-Reward PUCT search.

Provides protocols and implementations for recording a thin sidecar log
(trajectory.jsonl) alongside the existing checkpoint, one record per step.
Records are joinable with the final PuctCheckpoint archive via best_ulid.
"""

from pathlib import Path
from typing import Protocol

from loguru import logger
from pydantic import BaseModel
from ulid import ULID

from gpu_forecasters.hill_climbing.domain import Node, ObservationT


class TrajectoryRecord(BaseModel):
    """One per-step entry in trajectory.jsonl. Joinable with the final
    PuctCheckpoint archive via best_ulid."""

    step: int
    best_ulid: ULID
    best_reward: float | None
    archive_size: int
    # best_node_depth is optional for backward compatibility with older
    # trajectory.jsonl files that were written before this field existed.
    best_node_depth: int | None = None

    model_config = {"frozen": True, "arbitrary_types_allowed": True}


class TrajectoryProvider(Protocol[ObservationT]):
    def record(self, step: int, best_node: Node[ObservationT], archive_size: int) -> None: ...


class NoOpTrajectoryProvider:
    """Default. Used when a caller doesn't want trajectory logging."""

    def record(self, step: int, best_node: Node[ObservationT], archive_size: int) -> None:
        pass


class FileTrajectoryProvider:
    """Appends one TrajectoryRecord per step to a .jsonl file.

    Fault-tolerant: write failures are logged via loguru and swallowed —
    search must never die because a trajectory write failed.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, step: int, best_node: Node[ObservationT], archive_size: int) -> None:
        try:
            rec = TrajectoryRecord(
                step=step,
                best_ulid=best_node.ulid,
                best_reward=best_node.evaluation.reward,
                archive_size=archive_size,
                best_node_depth=len(best_node.ancestors),
            )
            with self.path.open("a") as f:
                _ = f.write(rec.model_dump_json() + "\n")
        except Exception as exc:
            logger.warning(f"Trajectory write to {self.path} failed: {exc}")


def load_trajectory(path: Path) -> list[TrajectoryRecord]:
    """Read and parse a trajectory.jsonl file.

    Canonical reader for trajectory files. Use this rather than rolling your
    own JSONL parser in experiment code — it guarantees consistent ULID
    handling across readers and avoids subtle `ULID == str` comparison bugs.

    Returns records in file order (i.e. step order). Returns an empty list
    if the file does not exist. Blank lines are skipped; malformed lines
    raise pydantic ValidationError.
    """
    if not path.exists():
        return []
    records: list[TrajectoryRecord] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(TrajectoryRecord.model_validate_json(line))
    return records
