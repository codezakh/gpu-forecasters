"""Distribution-shape utilities for v2 calibration."""

from __future__ import annotations

import math

from gpu_forecasters.landscape_map.v2 import SUCCESS_BINS, SpeedupBin


def uniform_distribution() -> dict[SpeedupBin, float]:
    """The uniform simplex over :data:`SUCCESS_BINS`.

    Used as the parse-failure fallback in the evaluator: a model that
    refuses (or fails) to answer is scored as if it had emitted the
    uniform distribution, which preserves the proper-scoring penalty.
    """
    p = 1.0 / len(SUCCESS_BINS)
    return {b: p for b in SUCCESS_BINS}


def entropy(distribution: dict[SpeedupBin, float]) -> float:
    """Shannon entropy of the distribution in nats."""
    return -sum(p * math.log(p) for p in distribution.values() if p > 0)
