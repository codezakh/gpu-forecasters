from __future__ import annotations

from typing import Protocol

from gpu_forecasters.ttt_discover.v2.domain.outcome import TriMulRLOutcome


class RewardScalarizer(Protocol):
    def scalarize(self, outcome: TriMulRLOutcome | None) -> float: ...
