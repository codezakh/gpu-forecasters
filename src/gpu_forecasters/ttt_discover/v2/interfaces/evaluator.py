from __future__ import annotations

from typing import Protocol

from arid_badger.ttt_discover.v2.domain.outcome import TriMulRLOutcome


class KernelEvaluator(Protocol):
    async def evaluate(self, code: str) -> TriMulRLOutcome: ...
