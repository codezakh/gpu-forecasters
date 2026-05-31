"""Durable, append-only log of ``SearchEvent``s.

Contract: ``append(event)`` does not return until the event has been
written and fsynced to disk. If the process dies immediately after the
call, the event is in the log. If it dies during the call, the
at-worst outcome is a truncated trailing line that ``read_all`` will
skip.

Per-event fsync is the default because it's correct without thought
and the workloads here are bound by LLM/GPU latency, not disk. If
fsync cost ever shows up in profiles, batch at step boundaries — the
event stream is already naturally fenced by ``StepCompleted``.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Generic, Protocol

from loguru import logger

from arid_badger.hill_climbing.domain import ObservationT
from arid_badger.max_reward_puct.v2.events import (
    SearchEvent,
    search_event_adapter,
)


class EventLog(Protocol[ObservationT]):
    """Append-only durable log of typed search events."""

    def append(self, event: SearchEvent[ObservationT]) -> None: ...

    def read_all(self) -> list[SearchEvent[ObservationT]]: ...


class InMemoryEventLog(Generic[ObservationT]):
    """Non-durable log for tests. Holds events in a list."""

    def __init__(self) -> None:
        self._events: list[SearchEvent[ObservationT]] = []

    def append(self, event: SearchEvent[ObservationT]) -> None:
        self._events.append(event)

    def read_all(self) -> list[SearchEvent[ObservationT]]:
        return list(self._events)


class FileEventLog(Generic[ObservationT]):
    """JSONL-backed event log with per-append fsync.

    One line per event, each line a JSON document matching the
    discriminated ``SearchEvent[ObservationT]`` union. Safe for a
    single writer process; no cross-process locking. In-process
    concurrent appends (from the driver's worker threads, if any)
    are serialized via an internal lock.
    """

    def __init__(
        self,
        path: Path,
        *,
        observation_type: type[ObservationT],
    ) -> None:
        """
        Args:
            path: log file location. Created if missing.
            observation_type: the concrete ``ObservationT`` for this
                search (e.g. ``NoFeedback``, ``TriMulObservation``).
                Used to build the discriminated ``TypeAdapter`` that
                drives JSON (de)serialization.
        """
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._adapter = search_event_adapter(observation_type)
        self._lock = threading.Lock()

    @property
    def path(self) -> Path:
        return self._path

    def append(self, event: SearchEvent[ObservationT]) -> None:
        line = self._adapter.dump_json(event).decode("utf-8")
        with self._lock:
            with open(self._path, "a", encoding="utf-8") as f:
                f.write(line)
                f.write("\n")
                f.flush()
                os.fsync(f.fileno())

    def read_all(self) -> list[SearchEvent[ObservationT]]:
        if not self._path.exists():
            return []
        events: list[SearchEvent[ObservationT]] = []
        with open(self._path, "r", encoding="utf-8") as f:
            for i, raw in enumerate(f, start=1):
                line = raw.strip()
                if not line:
                    continue
                try:
                    events.append(self._adapter.validate_json(line))
                except Exception as exc:
                    # A trailing line may be truncated by a crash
                    # mid-write. Tolerate that on the last non-empty
                    # line; a corrupt mid-file line is a real problem.
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
