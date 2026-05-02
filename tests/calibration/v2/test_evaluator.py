"""Invariant tests for the v2 calibration evaluator."""

from __future__ import annotations

import math

import pytest

from arid_badger.calibration.v2 import (
    CalibrationDatum,
    evaluate_calibration,
)
from arid_badger.landscape_map.v2 import (
    SUCCESS_BINS,
    KernelRuntimeEstimate,
    SpeedupBin,
)


def _estimate(
    bin_probabilities: dict[SpeedupBin, float],
    predicted_bin: SpeedupBin,
    raw_sum: float = 1.0,
) -> KernelRuntimeEstimate:
    return KernelRuntimeEstimate(
        predicted_bin=predicted_bin,
        bin_probabilities=bin_probabilities,
        reasoning="test",
        raw_probability_sum=raw_sum,
    )


def _one_hot(b: SpeedupBin) -> dict[SpeedupBin, float]:
    return {bb: 1.0 if bb == b else 0.0 for bb in SUCCESS_BINS}


def test_perfect_predictions_collapse_to_zero_loss_and_one_accuracy():
    data = [
        CalibrationDatum(
            true_bin=SpeedupBin.MINOR_SPEEDUP,
            estimate=_estimate(_one_hot(SpeedupBin.MINOR_SPEEDUP), SpeedupBin.MINOR_SPEEDUP),
        ),
        CalibrationDatum(
            true_bin=SpeedupBin.HIGH_SPEEDUP,
            estimate=_estimate(_one_hot(SpeedupBin.HIGH_SPEEDUP), SpeedupBin.HIGH_SPEEDUP),
        ),
    ]
    report = evaluate_calibration(data)
    assert report.accuracy == 1.0
    assert report.parsed_rate == 1.0
    assert report.mean_brier == pytest.approx(0.0, abs=1e-12)
    assert report.mean_crps == pytest.approx(0.0, abs=1e-12)
    assert report.mean_nll == pytest.approx(0.0, abs=1e-12)
    assert report.ece == 0.0


def test_parse_failures_get_uniform_fallback_penalty():
    """A model that refuses to answer is penalized as if uniform."""
    data = [
        CalibrationDatum(true_bin=SpeedupBin.MINOR_SPEEDUP, estimate=None)
    ]
    report = evaluate_calibration(data)
    assert report.parsed_rate == 0.0
    # Uniform NLL = log(8); evaluator penalizes the failure with that.
    assert report.mean_nll == pytest.approx(math.log(8), abs=1e-12)
    # Parsed-only metric stays at zero — there are no parsed rows.
    assert report.mean_nll_parsed == 0.0
    assert report.mean_raw_probability_sum is None


def test_ece_collapses_to_conf_minus_acc_when_all_in_one_bucket():
    """If all parsed rows fall in the same confidence bucket, ECE
    is exactly |mean_conf - accuracy|."""
    truth = SpeedupBin.MINOR_SPEEDUP
    other = SpeedupBin.HIGH_SPEEDUP
    # Three rows all at confidence 0.85 (bucket [0.8, 0.9)), one
    # correct, two wrong → accuracy 1/3, mean confidence 0.85.
    p = {b: 0.0 for b in SUCCESS_BINS}
    p[truth] = 0.85
    p[other] = 0.15

    correct = CalibrationDatum(
        true_bin=truth, estimate=_estimate(p, truth)
    )
    p_other = {b: 0.0 for b in SUCCESS_BINS}
    p_other[other] = 0.85
    p_other[truth] = 0.15
    wrong = CalibrationDatum(
        true_bin=truth, estimate=_estimate(p_other, other)
    )

    report = evaluate_calibration([correct, wrong, wrong])
    assert report.ece == pytest.approx(abs(0.85 - 1 / 3), abs=1e-12)
