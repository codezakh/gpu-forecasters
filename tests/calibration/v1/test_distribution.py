"""Tests for ``bin_distribution`` and ``entropy``."""

from __future__ import annotations

import math

import pytest

from arid_badger.calibration.v1 import (
    LikertNumericMapping,
    PREDICTED_BINS,
    bin_distribution,
    entropy,
    uniform_distribution,
)
from arid_badger.landscape_map.v1.domain import (
    KernelRuntimeEstimate,
    LikertConfidence,
    SpeedupBin,
)


def _estimate_with(levels: dict[SpeedupBin, LikertConfidence]) -> KernelRuntimeEstimate:
    full = {b: LikertConfidence.VERY_LOW for b in PREDICTED_BINS}
    full.update(levels)
    # predicted_bin is whichever bin has the highest verbalized level.
    predicted = max(
        full.items(),
        key=lambda item: list(LikertConfidence).index(item[1]),
    )[0]
    return KernelRuntimeEstimate(
        predicted_bin=predicted, bin_confidences=full, reasoning=""
    )


def test_bin_distribution_normalizes_to_one() -> None:
    est = _estimate_with(
        {
            SpeedupBin.MINOR_SLOWDOWN: LikertConfidence.HIGH,
            SpeedupBin.MINOR_SPEEDUP: LikertConfidence.MODERATE,
        }
    )
    dist = bin_distribution(est, LikertNumericMapping())
    assert math.isclose(sum(dist.values()), 1.0, abs_tol=1e-9)
    assert set(dist.keys()) == set(PREDICTED_BINS)


def test_bin_distribution_assigns_more_mass_to_higher_confidence() -> None:
    est = _estimate_with(
        {
            SpeedupBin.MINOR_SPEEDUP: LikertConfidence.VERY_HIGH,
            SpeedupBin.MINOR_SLOWDOWN: LikertConfidence.LOW,
        }
    )
    dist = bin_distribution(est, LikertNumericMapping())
    assert dist[SpeedupBin.MINOR_SPEEDUP] > dist[SpeedupBin.MINOR_SLOWDOWN]


def test_bin_distribution_falls_back_to_uniform_on_zero_total() -> None:
    # Mapping where every Likert level is 0 → degenerate input. We
    # don't expect this in practice but it must not divide-by-zero.
    zero_mapping = LikertNumericMapping(
        very_low=0.0, low=0.0, moderate=0.0, high=0.0, very_high=0.0
    )
    est = _estimate_with({SpeedupBin.MINOR_SPEEDUP: LikertConfidence.VERY_HIGH})
    dist = bin_distribution(est, zero_mapping)
    expected = 1.0 / len(PREDICTED_BINS)
    for v in dist.values():
        assert v == pytest.approx(expected)


def test_uniform_distribution_sums_to_one() -> None:
    dist = uniform_distribution()
    assert math.isclose(sum(dist.values()), 1.0, abs_tol=1e-9)
    assert all(v == pytest.approx(1.0 / 8) for v in dist.values())


def test_entropy_uniform_max() -> None:
    # ln(8) is the entropy of a uniform 8-class distribution in nats.
    assert entropy(uniform_distribution()) == pytest.approx(math.log(8))


def test_entropy_one_hot_zero() -> None:
    onehot = {b: 0.0 for b in PREDICTED_BINS}
    onehot[SpeedupBin.MINOR_SPEEDUP] = 1.0
    assert entropy(onehot) == pytest.approx(0.0)
