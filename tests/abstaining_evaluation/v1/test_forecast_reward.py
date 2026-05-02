"""Unit tests for ``ExpectedSpeedupReward``."""

from __future__ import annotations

import math

from arid_badger.abstaining_evaluation.v1.forecast_reward import (
    ExpectedSpeedupReward,
)
from arid_badger.landscape_map.v2 import (
    SUCCESS_BINS,
    KernelRuntimeEstimate,
    SpeedupBin,
)


def _estimate(probs: dict[SpeedupBin, float]) -> KernelRuntimeEstimate:
    if set(probs.keys()) != set(SUCCESS_BINS):
        raise AssertionError("test fixture must cover all SUCCESS_BINS")
    predicted = max(probs.items(), key=lambda kv: kv[1])[0]
    return KernelRuntimeEstimate(
        predicted_bin=predicted,
        bin_probabilities=probs,
        reasoning="test fixture",
        raw_probability_sum=1.0,
    )


def _delta_on(bin_: SpeedupBin) -> KernelRuntimeEstimate:
    """All probability mass on a single bin."""
    return _estimate({b: (1.0 if b == bin_ else 0.0) for b in SUCCESS_BINS})


def test_delta_on_minor_speedup_returns_geometric_midpoint() -> None:
    """Bin 5 (1.0x, 1.414x] geometric midpoint = 2 ** 0.375 ≈ 1.297."""
    reward = ExpectedSpeedupReward()(_delta_on(SpeedupBin.MINOR_SPEEDUP))
    assert math.isclose(reward, 2.0 ** 0.375, rel_tol=1e-9)


def test_delta_on_severe_slowdown_returns_open_bin_midpoint() -> None:
    """Bin 1 (≤0.25x) is open-ended low; uses representative finite midpoint."""
    reward = ExpectedSpeedupReward()(_delta_on(SpeedupBin.SEVERE_SLOWDOWN))
    assert math.isclose(reward, 2.0 ** -2.25, rel_tol=1e-9)
    # Sanity: should be positive but small.
    assert 0.0 < reward < 0.25


def test_delta_on_extreme_speedup_returns_open_bin_midpoint() -> None:
    """Bin 8 (>4.0x) is open-ended high; uses representative finite midpoint."""
    reward = ExpectedSpeedupReward()(_delta_on(SpeedupBin.EXTREME_SPEEDUP))
    assert math.isclose(reward, 2.0 ** 2.25, rel_tol=1e-9)
    # Sanity: bounded above 4.0x.
    assert reward > 4.0


def test_uniform_distribution_is_average_of_midpoints() -> None:
    """Σ p_b · midpoint(b) with p_b = 1/8 = mean of the eight midpoints."""
    p = 1.0 / 8.0
    estimate = _estimate({b: p for b in SUCCESS_BINS})
    reward = ExpectedSpeedupReward()(estimate)
    # Per the constants in forecast_reward.py:
    midpoints = [
        2.0 ** -2.25,    # bin 1
        2.0 ** -1.25,    # bin 2
        2.0 ** -0.625,   # bin 3
        2.0 ** -0.375,   # bin 4
        2.0 ** 0.375,    # bin 5
        2.0 ** 0.625,    # bin 6
        2.0 ** 1.25,     # bin 7
        2.0 ** 2.25,     # bin 8
    ]
    expected = sum(midpoints) / 8.0
    assert math.isclose(reward, expected, rel_tol=1e-9)


def test_two_bin_mixture_matches_weighted_sum() -> None:
    """0.7 mass on bin 5, 0.3 on bin 7 → 0.7 * 1.297 + 0.3 * 2.378."""
    probs = {b: 0.0 for b in SUCCESS_BINS}
    probs[SpeedupBin.MINOR_SPEEDUP] = 0.7
    probs[SpeedupBin.HIGH_SPEEDUP] = 0.3
    reward = ExpectedSpeedupReward()(_estimate(probs))
    expected = 0.7 * 2.0 ** 0.375 + 0.3 * 2.0 ** 1.25
    assert math.isclose(reward, expected, rel_tol=1e-9)


def test_reward_lies_in_range_of_midpoints_for_any_simplex() -> None:
    """A simplex over the eight bins must produce a reward in
    [min_midpoint, max_midpoint]."""
    # Pick an asymmetric distribution.
    probs = {
        SpeedupBin.SEVERE_SLOWDOWN:      0.1,
        SpeedupBin.SIGNIFICANT_SLOWDOWN: 0.05,
        SpeedupBin.MODERATE_SLOWDOWN:    0.05,
        SpeedupBin.MINOR_SLOWDOWN:       0.10,
        SpeedupBin.MINOR_SPEEDUP:        0.30,
        SpeedupBin.SIGNIFICANT_SPEEDUP:  0.20,
        SpeedupBin.HIGH_SPEEDUP:         0.15,
        SpeedupBin.EXTREME_SPEEDUP:      0.05,
    }
    reward = ExpectedSpeedupReward()(_estimate(probs))
    assert 2.0 ** -2.25 <= reward <= 2.0 ** 2.25
