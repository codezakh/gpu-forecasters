"""Tests for the three confidence-score implementations."""

from __future__ import annotations


import pytest

from arid_badger.abstaining_evaluation.v1 import (
    MaxProbScore,
    NegEntropyScore,
    Top2MarginScore,
)
from arid_badger.landscape_map.v2 import SUCCESS_BINS, SpeedupBin

from ._fixtures import make_estimate


def test_max_prob_returns_argmax_mass() -> None:
    e = make_estimate(predicted_bin=SpeedupBin.MINOR_SPEEDUP, confidence=0.7)
    assert MaxProbScore()(e) == pytest.approx(0.7)


def test_neg_entropy_one_for_delta_zero_for_uniform() -> None:
    delta = make_estimate(
        predicted_bin=SpeedupBin.MINOR_SPEEDUP, confidence=1.0
    )
    score = NegEntropyScore()(delta)
    assert score == pytest.approx(1.0)

    uniform_probs = {b: 1.0 / 8.0 for b in SUCCESS_BINS}
    from arid_badger.landscape_map.v2 import KernelRuntimeEstimate
    uniform = KernelRuntimeEstimate(
        predicted_bin=SpeedupBin.MINOR_SPEEDUP,
        bin_probabilities=uniform_probs,
        reasoning="x",
        raw_probability_sum=1.0,
    )
    assert NegEntropyScore()(uniform) == pytest.approx(0.0, abs=1e-9)


def test_top2_margin_is_p_top1_minus_p_top2() -> None:
    e = make_estimate(predicted_bin=SpeedupBin.MINOR_SPEEDUP, confidence=0.5)
    # 7 other bins each get 0.5/7. Top-2 margin = 0.5 - 0.5/7.
    expected = 0.5 - 0.5 / 7
    assert Top2MarginScore()(e) == pytest.approx(expected, abs=1e-9)


def test_max_prob_strictly_below_1_for_non_delta() -> None:
    e = make_estimate(predicted_bin=SpeedupBin.MINOR_SPEEDUP, confidence=0.5)
    assert MaxProbScore()(e) < 1.0
    assert MaxProbScore()(e) > 1.0 / 8.0
