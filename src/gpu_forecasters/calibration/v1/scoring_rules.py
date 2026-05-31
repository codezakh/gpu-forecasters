"""Per-row proper scoring rules over the 8-bin distribution.

Both ``brier`` and ``crps`` operate on a normalized distribution over
``PREDICTED_BINS`` and a ground-truth ``SpeedupBin``. They do not
parse the raw ``KernelRuntimeEstimate`` — callers are responsible for
calling ``bin_distribution`` first (or ``uniform_distribution`` as a
parse-failure fallback). This keeps the scoring rules independent of
the verbalized Likert convention.

Both functions return values in ``[0, 1]`` so they can be combined
with the existing distance reward without renormalization. CRPS in
particular is normalized by ``K - 1 = 7`` which is the maximum
possible value of the un-normalized sum (true bin at one end, all
mass at the other).
"""

from __future__ import annotations

from gpu_forecasters.landscape_map.v1.domain import SpeedupBin

from .domain import PREDICTED_BINS


def brier(distribution: dict[SpeedupBin, float], true_bin: SpeedupBin) -> float:
    """Brier score over the 8 bins treated as nominal.

    ``B = 0.5 * Σ_k (p_k - y_k)²`` where ``y`` is one-hot at
    ``true_bin``. The factor of 0.5 normalizes the maximum value
    (mass concentrated entirely on a wrong bin → ``2`` un-normalized,
    ``1`` after the factor) so the metric lives in ``[0, 1]``,
    matching CRPS's range. Lower is better.
    """
    if true_bin not in PREDICTED_BINS:
        raise ValueError(
            f"true_bin must be a predicted bin (1..8); got {true_bin!r}"
        )
    raw = sum(
        (distribution[b] - (1.0 if b == true_bin else 0.0)) ** 2
        for b in PREDICTED_BINS
    )
    return 0.5 * raw


def crps(distribution: dict[SpeedupBin, float], true_bin: SpeedupBin) -> float:
    """CRPS over the ordinal CDF, normalized to ``[0, 1]``.

    Lower is better. We treat the bins as ordered 1..8 and compute the
    discrete CRPS:

        CRPS = Σ_{k=1}^{K-1} (F_k - F*_k)²

    where ``F`` is the predicted CDF and ``F*`` is the true CDF
    (Heaviside step at ``true_bin``). The sum runs to ``K - 1 = 7``
    because ``F_K = F*_K = 1`` always.

    The maximum possible un-normalized CRPS occurs when the true bin
    is at one end (1 or 8) and all predicted mass is at the other.
    In that case every term is 1 and the sum is 7. Dividing by 7
    rescales the metric to ``[0, 1]``.

    The reward shape used by ``BlendedDistanceCRPSReward`` is
    ``r_calib = 1 - crps(...)`` — likewise in ``[0, 1]``, with higher
    being better.
    """
    if true_bin not in PREDICTED_BINS:
        raise ValueError(
            f"true_bin must be a predicted bin (1..8); got {true_bin!r}"
        )
    cumulative_pred = 0.0
    cumulative_true = 0.0
    raw = 0.0
    # Iterate bins 1..7; bin 8's contribution is always (1 - 1)^2 = 0.
    for b in PREDICTED_BINS[:-1]:
        cumulative_pred += distribution[b]
        if b >= true_bin:
            cumulative_true = 1.0
        raw += (cumulative_pred - cumulative_true) ** 2
    return raw / (len(PREDICTED_BINS) - 1)


def crps_calibration_reward(
    distribution: dict[SpeedupBin, float], true_bin: SpeedupBin
) -> float:
    """Convenience wrapper used by training-time reward shaping.

    Returns ``1 - crps(...)`` so the value lives in ``[0, 1]`` with
    higher = better, matching the convention of the existing
    distance-shaped reward. RL trainers can blend this with the
    distance reward without further renormalization.
    """
    return 1.0 - crps(distribution, true_bin)
