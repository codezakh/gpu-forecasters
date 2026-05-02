"""Map a surrogate forecast to a search-side reward.

The v2 max-reward PUCT search treats ``Evaluation.reward`` as a single
scalar that is comparable across nodes. The real evaluator produces
``SuccessFeedback.aggregated_speedup`` (a continuous speedup ratio);
the surrogate produces a probability distribution over speedup bins.
A ``ForecastRewardPolicy`` is the seam that turns the second into a
number that is comparable with the first.

We ship one concrete (``ExpectedSpeedupReward``); the protocol is here
because callers that want to swap in an alternative reward extractor
(argmax-bin midpoint, expected speedup with bin-1 zeroed out, ...)
should not have to fork the compound provider.
"""

from __future__ import annotations

import math
from typing import Protocol

from arid_badger.landscape_map.v2 import (
    SUCCESS_BINS,
    KernelRuntimeEstimate,
    SpeedupBin,
)


class ForecastRewardPolicy(Protocol):
    """Maps a forecast to a real-eval-comparable reward."""

    def __call__(self, estimate: KernelRuntimeEstimate) -> float: ...


# Bin boundaries: floor(2 * log2(S)) + 4 in [1, 8] (see SpeedupBin).
# Bin b has interval (2 ** ((b - 5) / 2), 2 ** ((b - 4) / 2)] for b in
# [2, 7]; bins 1 and 8 are open on the slow / fast side respectively.
# Geometric midpoint of an interval (lo, hi] in log2 space is
# 2 ** ((log2(lo) + log2(hi)) / 2).
_GEOMETRIC_BIN_MIDPOINTS: dict[SpeedupBin, float] = {
    SpeedupBin.SIGNIFICANT_SLOWDOWN: 2.0 ** -1.25,    # (0.25, 0.5]
    SpeedupBin.MODERATE_SLOWDOWN:    2.0 ** -0.625,   # (0.5, 0.707]
    SpeedupBin.MINOR_SLOWDOWN:       2.0 ** -0.375,   # (0.707, 1.0]
    SpeedupBin.MINOR_SPEEDUP:        2.0 ** 0.375,    # (1.0, 1.414]
    SpeedupBin.SIGNIFICANT_SPEEDUP:  2.0 ** 0.625,    # (1.414, 2.0]
    SpeedupBin.HIGH_SPEEDUP:         2.0 ** 1.25,     # (2.0, 4.0]
}


# Bins 1 and 8 are open-ended; we pick representative finite midpoints
# rather than 0 or +inf so the expected-speedup integral stays bounded
# and the reward stays in a usable range. The values below are the
# geometric midpoint of the bin's *closed* edge with one further
# half-octave step on the open side, matching how the speedup-distance
# metric in ``risks.py`` would handle them in absence of empirical data.
_BIN_1_OPEN_MIDPOINT: float = 2.0 ** -2.25   # (-, 0.25]; representative ~0.21
_BIN_8_OPEN_MIDPOINT: float = 2.0 ** 2.25    # (4.0, +); representative ~4.76


def _midpoint(bin_: SpeedupBin) -> float:
    if bin_ == SpeedupBin.SEVERE_SLOWDOWN:
        return _BIN_1_OPEN_MIDPOINT
    if bin_ == SpeedupBin.EXTREME_SPEEDUP:
        return _BIN_8_OPEN_MIDPOINT
    return _GEOMETRIC_BIN_MIDPOINTS[bin_]


class ExpectedSpeedupReward:
    """Reward = Σ p_b · midpoint(b) over the eight success bins.

    Bins 2-7 use the geometric midpoint of their closed interval. Bins
    1 and 8 use a representative finite midpoint one half-octave past
    the bin's closed edge, so the expected value stays bounded in a
    range commensurate with what the real evaluator returns.
    """

    def __call__(self, estimate: KernelRuntimeEstimate) -> float:
        total = 0.0
        for bin_ in SUCCESS_BINS:
            total += estimate.bin_probabilities[bin_] * _midpoint(bin_)
        if not math.isfinite(total):
            raise ValueError(
                f"expected speedup is not finite: got {total!r} "
                f"from distribution {estimate.bin_probabilities!r}"
            )
        return total


__all__ = [
    "ExpectedSpeedupReward",
    "ForecastRewardPolicy",
]
