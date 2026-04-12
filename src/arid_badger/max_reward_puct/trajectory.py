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

from arid_badger.hill_climbing.domain import Node, ObservationT


class TrajectoryRecord(BaseModel):
    """One per-step entry in trajectory.jsonl. Joinable with the final
    PuctCheckpoint archive via best_ulid."""

    step: int
    best_ulid: ULID
    best_reward: float | None
    archive_size: int

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
            )
            with self.path.open("a") as f:
                _ = f.write(rec.model_dump_json() + "\n")
        except Exception as exc:
            logger.warning(f"Trajectory write to {self.path} failed: {exc}")
