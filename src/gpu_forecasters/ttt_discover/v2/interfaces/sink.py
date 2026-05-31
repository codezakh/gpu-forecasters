from __future__ import annotations

from typing import Protocol

from gpu_forecasters.ttt_discover.v2.domain.records import RolloutRecord


class RolloutSink(Protocol):
    def record(self, record: RolloutRecord) -> None: ...
