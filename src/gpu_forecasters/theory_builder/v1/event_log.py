"""Durable, append-only log of ``TheoryEvent``s.

Same shape as ``gpu_forecasters.max_reward_puct.v2.event_log``: per-event
fsync, JSONL on disk, single-writer, in-process appends serialised by
an internal lock. Truncated trailing line on read is tolerated;
corrupt mid-file lines raise.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Generic, Protocol

from loguru import logger

from gpu_forecasters.hill_climbing.domain import ObservationT
from gpu_forecasters.theory_builder.v1.events import (
    TheoryEvent,
    theory_event_adapter,
)


class TheoryEventLog(Protocol[ObservationT]):
    """Append-only durable log of typed theory-builder events."""

    def append(self, event: TheoryEvent[ObservationT]) -> None: ...

    def read_all(self) -> list[TheoryEvent[ObservationT]]: ...


class InMemoryTheoryEventLog(Generic[ObservationT]):
    """Non-durable log for tests."""

    def __init__(self) -> None:
        self._events: list[TheoryEvent[ObservationT]] = []

    def append(self, event: TheoryEvent[ObservationT]) -> None:
        self._events.append(event)

    def read_all(self) -> list[TheoryEvent[ObservationT]]:
        return list(self._events)


class FileTheoryEventLog(Generic[ObservationT]):
    """JSONL-backed event log with per-append fsync."""

    def __init__(
        self,
        path: Path,
        *,
        observation_type: type[ObservationT],
    ) -> None:
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._adapter = theory_event_adapter(observation_type)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def append(self, event: TheoryEvent[ObservationT]) -> None:
        line = self._adapter.dump_json(event).decode("utf-8")
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())

    def read_all(self) -> list[TheoryEvent[ObservationT]]:
        if not self._path.exists():
            return []
        events: list[TheoryEvent[ObservationT]] = []
        with open(self._path, "r", encoding="utf-8") as f:
            for i, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    events.append(self._adapter.validate_json(line))
                except Exception as exc:
                    remaining = f.read().strip()
                    if remaining:
                        raise ValueError(
                            f"Corrupt non-final line {i} in {self._path}"
                        ) from exc
                    logger.warning(
                        "Dropping truncated trailing line {i} from {path}: {exc}",
                        i=i,
                        path=self._path,
                        exc=exc,
                    )
                    break
        return events


__all__ = [
    "TheoryEventLog",
    "InMemoryTheoryEventLog",
    "FileTheoryEventLog",
]
