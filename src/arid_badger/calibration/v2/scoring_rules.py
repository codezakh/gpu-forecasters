"""Pure proper-scoring-rule functions over v2 simplex distributions.

Each function takes a ``dict[SpeedupBin, float]`` distribution (assumed
non-negative and normalized to sum to 1; the v2 parser guarantees
this) and a true bin. They operate over exactly the
:data:`SUCCESS_BINS` keys.
"""

from __future__ import annotations

import math

from arid_badger.landscape_map.v2 import SUCCESS_BINS, SpeedupBin


# Smallest probability we score for NLL, to keep
# log(0) from blowing up. Mirrors typical floor used in CE losses.
_NLL_FLOOR = 1e-12


def brier(distribution: dict[SpeedupBin, float], true_bin: SpeedupBin) -> float:
    """Half-Brier (mean squared error) of the distribution against the one-hot.

    Lower is better. Range ``[0, 1]`` for distributions and one-hots
    over ``SUCCESS_BINS``.
    """
    raw = sum(
        (distribution[b] - (1.0 if b == true_bin else 0.0)) ** 2
        for b in SUCCESS_BINS
    )
    return 0.5 * raw


def crps(distribution: dict[SpeedupBin, float], true_bin: SpeedupBin) -> float:
    """Continuous Ranked Probability Score over the eight ordinal bins.

    Sum of squared CDF gaps over the first ``K - 1 = 7`` bins,
    normalized by ``K - 1``. Lower is better. Range ``[0, 1]``.
    """
    cumulative_pred = 0.0
    cumulative_true = 0.0
    raw = 0.0
    for b in SUCCESS_BINS[:-1]:
        cumulative_pred += distribution[b]
        if b >= true_bin:
            cumulative_true = 1.0
        raw += (cumulative_pred - cumulative_true) ** 2
    return raw / (len(SUCCESS_BINS) - 1)


def nll(distribution: dict[SpeedupBin, float], true_bin: SpeedupBin) -> float:
    """Negative log-likelihood of the true bin under the distribution.

    Lower is better. Range ``[0, +inf)`` — confidently-wrong predictions
    grow without bound, which is the asymmetry that makes NLL a
    sharper calibration penalty than Brier or CRPS.
    """
    p = distribution[true_bin]
    return -math.log(max(p, _NLL_FLOOR))
