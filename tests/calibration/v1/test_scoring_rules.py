"""Tests for the per-row Brier and CRPS scoring rules.

Pure-function unit tests over hand-constructed distributions. The
goal is to pin down the corner cases where it's easy to get the math
wrong: perfect prediction, all-mass-on-wrong-bin, ordinal vs nominal
sensitivity.
"""

from __future__ import annotations

import math

import pytest

from arid_badger.calibration.v1 import PREDICTED_BINS, brier, crps
from arid_badger.landscape_map.v1.domain import SpeedupBin


def _onehot(target: SpeedupBin) -> dict[SpeedupBin, float]:
    return {b: (1.0 if b == target else 0.0) for b in PREDICTED_BINS}


def _uniform() -> dict[SpeedupBin, float]:
    return {b: 1.0 / 8 for b in PREDICTED_BINS}


def test_brier_zero_on_perfect_prediction() -> None:
    for target in PREDICTED_BINS:
        assert brier(_onehot(target), target) == pytest.approx(0.0)


def test_crps_zero_on_perfect_prediction() -> None:
    for target in PREDICTED_BINS:
        assert crps(_onehot(target), target) == pytest.approx(0.0)


def test_brier_one_on_orthogonal_misprediction() -> None:
    # All mass on a wrong bin: Brier sum is (1)^2 + (1)^2 = 2,
    # times the 0.5 normalizer = 1.0.
    pred = _onehot(SpeedupBin.SEVERE_SLOWDOWN)
    assert brier(pred, SpeedupBin.EXTREME_SPEEDUP) == pytest.approx(1.0)


def test_crps_one_on_extreme_misprediction() -> None:
    # True bin = 8, all mass at bin 1: every CDF term has predicted
    # CDF = 1 from k=1..7 and true CDF = 0 from k=1..7, so each term
    # is 1 and the sum is 7. After dividing by 7, CRPS = 1.0.
    pred = _onehot(SpeedupBin.SEVERE_SLOWDOWN)
    assert crps(pred, SpeedupBin.EXTREME_SPEEDUP) == pytest.approx(1.0)


def test_crps_rewards_near_misses_more_than_brier() -> None:
    # All mass on bin 5 vs all mass on bin 4: nominal Brier sees them
    # as equally wrong (against true bin 4), but CRPS should rate the
    # near miss as substantially better than far misses.
    true = SpeedupBin.MINOR_SLOWDOWN
    pred_near = _onehot(SpeedupBin.MINOR_SPEEDUP)
    pred_far = _onehot(SpeedupBin.EXTREME_SPEEDUP)

    crps_near = crps(pred_near, true)
    crps_far = crps(pred_far, true)
    assert crps_near < crps_far

    # Brier *would* be identical for both — both miss the one-hot by
    # the same total mass. Pin this down to document the ordinal vs
    # nominal distinction the test is asserting.
    assert brier(pred_near, true) == pytest.approx(brier(pred_far, true))


def test_crps_uniform_against_extreme_truth() -> None:
    # Uniform distribution against true bin 1: predicted CDF goes
    # 1/8, 2/8, ..., 7/8 over k=1..7; true CDF is 1 for all k. Each
    # squared diff is (1 - k/8)^2; sum / 7 should land where the
    # closed form predicts.
    expected = sum((1 - k / 8) ** 2 for k in range(1, 8)) / 7
    assert crps(_uniform(), SpeedupBin.SEVERE_SLOWDOWN) == pytest.approx(expected)


def test_brier_uniform() -> None:
    # Uniform against true bin 4: 1 entry at (1 - 1/8)^2, 7 entries
    # at (1/8)^2; halved.
    raw = (1 - 1 / 8) ** 2 + 7 * (1 / 8) ** 2
    assert brier(_uniform(), SpeedupBin.MINOR_SLOWDOWN) == pytest.approx(0.5 * raw)


def test_crps_in_unit_interval() -> None:
    # Random-ish distributions; verify the metric stays in [0, 1].
    for true_bin in PREDICTED_BINS:
        for spike in PREDICTED_BINS:
            d = _onehot(spike)
            v = crps(d, true_bin)
            assert 0.0 <= v <= 1.0 + 1e-9


def test_brier_rejects_failure_truth() -> None:
    with pytest.raises(ValueError):
        _ = brier(_onehot(SpeedupBin.MINOR_SLOWDOWN), SpeedupBin.FAILURE)


def test_crps_rejects_failure_truth() -> None:
    with pytest.raises(ValueError):
        _ = crps(_onehot(SpeedupBin.MINOR_SLOWDOWN), SpeedupBin.FAILURE)


def test_crps_symmetric_in_distance() -> None:
    # Sanity: CRPS depends only on |bin gap| for one-hot predictions.
    one = _onehot(SpeedupBin.MODERATE_SLOWDOWN)
    two = _onehot(SpeedupBin.SIGNIFICANT_SLOWDOWN)
    # gap of 1 in either direction
    a = crps(one, SpeedupBin.SIGNIFICANT_SLOWDOWN)
    b = crps(two, SpeedupBin.MODERATE_SLOWDOWN)
    assert a == pytest.approx(b)


def test_distribution_must_be_normalized_for_crps_at_one() -> None:
    # Spot-check that the K-1 normalizer yields exactly 1.0 in the
    # worst-case extreme-misprediction setup. Guards against accidentally
    # dividing by K=8 instead of K-1=7.
    pred = _onehot(SpeedupBin.SEVERE_SLOWDOWN)
    assert math.isclose(crps(pred, SpeedupBin.EXTREME_SPEEDUP), 1.0, abs_tol=1e-12)
