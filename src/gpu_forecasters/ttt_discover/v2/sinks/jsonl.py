"""Append-only JSONL rollout sink + an in-memory list sink for tests."""

from __future__ import annotations

import threading
from io import TextIOWrapper
from pathlib import Path

from arid_badger.ttt_discover.v2.domain.records import RolloutRecord
from arid_badger.ttt_discover.v2.interfaces.sink import RolloutSink
from arid_badger.typing_utils import implements


class JsonlRolloutSink:
    _path: Path
    _handle: TextIOWrapper | None
    _lock: threading.Lock

    def __init__(self, path: Path) -> None:
        self._path = path
        self._handle = None
        self._lock = threading.Lock()

    def __deepcopy__(self, memo: dict[int, object]) -> "JsonlRolloutSink":
        # File handle + lock are not deepcopy-able; the sink is shared
        # mutable state and callers that deepcopy the surrounding config
        # (e.g. chz.asdict for hparam logging) should get identity back.
        return self

    def _ensure_handle(self) -> TextIOWrapper:
        if self._handle is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = self._path.open("a", encoding="utf-8")
        return self._handle

    def record(self, record: RolloutRecord) -> None:
        line = record.model_dump_json() + "\n"
        with self._lock:
            handle = self._ensure_handle()
            _ = handle.write(line)
            handle.flush()

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                self._handle.close()
                self._handle = None


class ListRolloutSink:
    records: list[RolloutRecord]

    def __init__(self) -> None:
        self.records = []

    def record(self, record: RolloutRecord) -> None:
        self.records.append(record)


_ = implements(RolloutSink)(JsonlRolloutSink)
_ = implements(RolloutSink)(ListRolloutSink)
