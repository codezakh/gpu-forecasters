"""Project a verbalized Likert distribution onto the 8-bin probability simplex.

The surrogate emits ``bin_confidences: dict[SpeedupBin, LikertConfidence]``
with one entry per bin in ``PREDICTED_BINS``. ``bin_distribution``
looks up each level in the numeric mapping and renormalizes so the
result is a proper probability distribution over the eight predicted
bins.

A degenerate input — every bin marked ``VERY_LOW`` and the mapping
sets ``very_low = 0`` — would produce an all-zero vector with no
defined renormalization. Surrogates we evaluate here always emit at
least one non-VERY_LOW level (the predicted bin); we still defend
against the edge case by falling back to uniform.
"""

from __future__ import annotations

import math

from gpu_forecasters.landscape_map.v1.domain import KernelRuntimeEstimate, SpeedupBin

from .domain import PREDICTED_BINS, LikertNumericMapping


def bin_distribution(
    estimate: KernelRuntimeEstimate, mapping: LikertNumericMapping
) -> dict[SpeedupBin, float]:
    """Map verbalized Likert per bin to a normalized probability per bin.

    Always returns a dict over the eight ``PREDICTED_BINS``. If the
    summed numeric weight is zero (every bin VERY_LOW + zero mapping),
    falls back to a uniform distribution rather than dividing by zero.
    """
    raw: dict[SpeedupBin, float] = {
        b: mapping.numeric_for(estimate.bin_confidences[b]) for b in PREDICTED_BINS
    }
    total = sum(raw.values())
    if total <= 0:
        uniform = 1.0 / len(PREDICTED_BINS)
        return {b: uniform for b in PREDICTED_BINS}
    return {b: w / total for b, w in raw.items()}


def uniform_distribution() -> dict[SpeedupBin, float]:
    """The fallback distribution applied to parse failures.

    Uniform over the eight predicted bins. Any single answer is one
    eighth, so a parse failure on any true bin contributes a fixed,
    non-trivial CRPS / Brier penalty.
    """
    uniform = 1.0 / len(PREDICTED_BINS)
    return {b: uniform for b in PREDICTED_BINS}


def entropy(distribution: dict[SpeedupBin, float]) -> float:
    """Shannon entropy of the bin distribution, in nats.

    Uses ``ln`` rather than ``log2`` since this is reported as a
    relative quantity (mean entropy across rows, compared between
    models); the unit choice is consistent within a single report.
    Zero-probability bins are skipped to avoid ``log(0)``.
    """
    return -sum(p * math.log(p) for p in distribution.values() if p > 0)
