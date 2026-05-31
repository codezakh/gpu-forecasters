"""Reward scalarizer that maps a ``SuccessFeedback`` onto ``target_us /
candidate_geomean_us``.

Failures of any kind (parse, compile, runtime, incorrect, infra) and the
``None`` cold-start outcome all map to ``0.0``. ``None`` is a
cold-start; the other failure variants signal "this rollout produced no
useful child" — they receive zero reward so the group mean baseline for
the surviving successful rollouts stays monotone in runtime.

Candidate geomean is taken over the per-case ``runtime_ns`` field of
``SuccessFeedback.per_case_speedups``; this mirrors how TriMul
leaderboards elsewhere in the repo aggregate.
"""

from __future__ import annotations

import math

from gpu_forecasters.trimul.core import SuccessFeedback
from gpu_forecasters.ttt_discover.v2.domain.outcome import TriMulRLOutcome
from gpu_forecasters.ttt_discover.v2.interfaces.scalarizer import RewardScalarizer
from gpu_forecasters.typing_utils import implements


class ScaleByTargetUs:
    _target_us: float

    def __init__(self, target_us: float) -> None:
        if target_us <= 0:
            raise ValueError("target_us must be positive")
        self._target_us = target_us

    def scalarize(self, outcome: TriMulRLOutcome | None) -> float:
        if outcome is None:
            return 0.0
        if not isinstance(outcome, SuccessFeedback):
            return 0.0
        if not outcome.per_case_speedups:
            return 0.0
        log_sum = 0.0
        for case in outcome.per_case_speedups:
            if case.runtime_ns <= 0:
                return 0.0
            log_sum += math.log(case.runtime_ns)
        geomean_ns = math.exp(log_sum / len(outcome.per_case_speedups))
        geomean_us = geomean_ns / 1_000.0
        return self._target_us / geomean_us


_ = implements(RewardScalarizer)(ScaleByTargetUs)
