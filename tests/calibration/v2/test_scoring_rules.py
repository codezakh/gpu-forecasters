"""Invariant tests for v2 scoring rules.

We test the *invariants that define correctness* — perfect prediction
scores zero on each rule, uniform distribution sits at the
neutral-prior value, NLL grows unbounded as the truth probability
collapses. We do not test pydantic round-tripping or framework
behavior.
"""

from __future__ import annotations

import math

import pytest

from gpu_forecasters.calibration.v2 import (
    brier,
    crps,
    nll,
    uniform_distribution,
)
from gpu_forecasters.landscape_map.v2 import SUCCESS_BINS, SpeedupBin


def _one_hot(true_bin: SpeedupBin) -> dict[SpeedupBin, float]:
    return {b: 1.0 if b == true_bin else 0.0 for b in SUCCESS_BINS}


def test_perfect_prediction_zero_brier_crps_nll():
    truth = SpeedupBin.MINOR_SPEEDUP
    p = _one_hot(truth)
    assert brier(p, truth) == 0.0
    assert crps(p, truth) == 0.0
    # log(1) == 0
    assert nll(p, truth) == pytest.approx(0.0, abs=1e-12)


def test_uniform_distribution_neutral_values():
    truth = SpeedupBin.MINOR_SPEEDUP
    p = uniform_distribution()
    # half-Brier on a uniform distribution against any one-hot:
    # (1/8 - 1)^2 + 7*(1/8)^2, all halved.
    expected_brier = 0.5 * ((1.0 - 1 / 8) ** 2 + 7 * (1 / 8) ** 2)
    assert brier(p, truth) == pytest.approx(expected_brier, abs=1e-12)
    # Uniform NLL is -log(1/8) regardless of truth bin.
    assert nll(p, truth) == pytest.approx(math.log(8), abs=1e-12)
    # Uniform CRPS depends on truth-bin position; just verify it's > 0
    # and less than 1.
    assert 0 < crps(p, truth) < 1


def test_nll_grows_unbounded_as_truth_probability_collapses():
    truth = SpeedupBin.MINOR_SPEEDUP
    other = SpeedupBin.HIGH_SPEEDUP
    # Very confident in the wrong answer.
    p = {b: 1e-9 if b != other else 1.0 - 7 * 1e-9 for b in SUCCESS_BINS}
    # NLL on truth ≈ -log(1e-9) ≈ 20.7
    score = nll(p, truth)
    assert score > 15.0


def test_crps_off_by_one_smaller_than_off_by_far():
    """The point of CRPS over Brier — ordinal closeness matters."""
    truth = SpeedupBin.MINOR_SPEEDUP  # bin 5
    near_miss = _one_hot(SpeedupBin.SIGNIFICANT_SPEEDUP)  # bin 6
    far_miss = _one_hot(SpeedupBin.SEVERE_SLOWDOWN)  # bin 1
    assert crps(near_miss, truth) < crps(far_miss, truth)
    # Brier doesn't have this property — both miss equally.
    assert brier(near_miss, truth) == brier(far_miss, truth)
