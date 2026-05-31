"""Tests for ``BinaryMismatchRisk``, ``SpeedupDistanceRisk``, ``RegretRisk``."""

from __future__ import annotations

import math

import pytest

from gpu_forecasters.abstaining_evaluation.v1 import (
    AbstainDecision,
    BinaryMismatchRisk,
    Predict,
    RegretRisk,
    SpeedupDistanceRisk,
    bin_midpoint,
)
from gpu_forecasters.landscape_map.v2 import SpeedupBin

from ._fixtures import make_comparison, make_estimate, true_bin_v2


def test_binary_mismatch_returns_nan_when_no_predictions() -> None:
    comparisons = [make_comparison(source_id="a", speedup=1.5)]
    decisions = [AbstainDecision()]
    assert math.isnan(BinaryMismatchRisk()(decisions, comparisons))


def test_binary_mismatch_one_correct_one_wrong() -> None:
    c1 = make_comparison(source_id="a", speedup=1.5)
    c2 = make_comparison(source_id="b", speedup=1.5)
    correct_bin = true_bin_v2(c1)
    wrong_bin = SpeedupBin.EXTREME_SPEEDUP
    assert wrong_bin != correct_bin
    decisions = [
        Predict(
            estimate=make_estimate(predicted_bin=correct_bin, confidence=0.8)
        ),
        Predict(
            estimate=make_estimate(predicted_bin=wrong_bin, confidence=0.8)
        ),
    ]
    assert BinaryMismatchRisk()(decisions, [c1, c2]) == pytest.approx(0.5)


def test_speedup_distance_uses_bin_midpoint() -> None:
    midpoint = 2.0 ** 0.25
    comparisons = [make_comparison(source_id="a", speedup=1.5)]
    correct_bin = true_bin_v2(comparisons[0])
    assert correct_bin == SpeedupBin.MINOR_SPEEDUP
    decisions = [
        Predict(
            estimate=make_estimate(predicted_bin=correct_bin, confidence=0.8)
        )
    ]
    assert SpeedupDistanceRisk()(decisions, comparisons) == pytest.approx(
        abs(midpoint - 1.5), abs=1e-9
    )


def test_regret_zero_when_top_predicted_bin_contains_pack_best() -> None:
    best_c = make_comparison(source_id="best", speedup=3.0)
    meh_c = make_comparison(source_id="meh", speedup=1.5)
    best_bin = true_bin_v2(best_c)
    meh_bin = true_bin_v2(meh_c)
    assert best_bin > meh_bin
    decisions = [
        Predict(estimate=make_estimate(predicted_bin=best_bin, confidence=0.9)),
        Predict(estimate=make_estimate(predicted_bin=meh_bin, confidence=0.9)),
    ]
    assert RegretRisk()(decisions, [best_c, meh_c]) == pytest.approx(
        0.0, abs=1e-9
    )


def test_regret_recovered_via_abstaining_real_eval() -> None:
    best_c = make_comparison(source_id="best", speedup=3.0)
    meh_c = make_comparison(source_id="meh", speedup=1.5)
    meh_bin = true_bin_v2(meh_c)
    decisions = [
        AbstainDecision(),
        Predict(estimate=make_estimate(predicted_bin=meh_bin, confidence=0.9)),
    ]
    assert RegretRisk()(decisions, [best_c, meh_c]) == pytest.approx(
        0.0, abs=1e-9
    )


def test_regret_pays_when_model_misses_best_and_we_didnt_abstain() -> None:
    best_c = make_comparison(source_id="best", speedup=3.0)
    meh_c = make_comparison(source_id="meh", speedup=1.5)
    meh_bin = true_bin_v2(meh_c)
    decisions = [
        Predict(
            estimate=make_estimate(
                predicted_bin=SpeedupBin.MODERATE_SLOWDOWN, confidence=0.9
            )
        ),
        Predict(estimate=make_estimate(predicted_bin=meh_bin, confidence=0.9)),
    ]
    assert RegretRisk()(decisions, [best_c, meh_c]) == pytest.approx(
        3.0 - 1.5
    )


def test_bin_midpoint_failure_bin_is_zero() -> None:
    assert (
        bin_midpoint(
            SpeedupBin.FAILURE,
            bin_1_empirical=0.1,
            bin_8_empirical=5.0,
        )
        == 0.0
    )


def test_bin_midpoint_open_bins_use_empirical() -> None:
    assert (
        bin_midpoint(
            SpeedupBin.SEVERE_SLOWDOWN,
            bin_1_empirical=0.1,
            bin_8_empirical=5.0,
        )
        == 0.1
    )
    assert (
        bin_midpoint(
            SpeedupBin.EXTREME_SPEEDUP,
            bin_1_empirical=0.1,
            bin_8_empirical=5.0,
        )
        == 5.0
    )
