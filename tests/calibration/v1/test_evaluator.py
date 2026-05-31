"""End-to-end tests of ``evaluate_calibration`` over hand-built data."""

from __future__ import annotations

import pytest

from gpu_forecasters.calibration.v1 import (
    CalibrationDatum,
    LikertNumericMapping,
    PREDICTED_BINS,
    evaluate_calibration,
)
from gpu_forecasters.landscape_map.v1.domain import (
    KernelRuntimeEstimate,
    LikertConfidence,
    SpeedupBin,
)


def _confident_correct_estimate(true_bin: SpeedupBin) -> KernelRuntimeEstimate:
    levels = {b: LikertConfidence.VERY_LOW for b in PREDICTED_BINS}
    levels[true_bin] = LikertConfidence.VERY_HIGH
    return KernelRuntimeEstimate(
        predicted_bin=true_bin, bin_confidences=levels, reasoning=""
    )


def _confident_wrong_estimate(predicted: SpeedupBin) -> KernelRuntimeEstimate:
    levels = {b: LikertConfidence.VERY_LOW for b in PREDICTED_BINS}
    levels[predicted] = LikertConfidence.VERY_HIGH
    return KernelRuntimeEstimate(
        predicted_bin=predicted, bin_confidences=levels, reasoning=""
    )


def test_perfect_predictions_give_low_metrics() -> None:
    data = [
        CalibrationDatum(
            true_bin=b, estimate=_confident_correct_estimate(b)
        )
        for b in PREDICTED_BINS
    ]
    report = evaluate_calibration(data)
    assert report.accuracy == pytest.approx(1.0)
    assert report.parsed_rate == pytest.approx(1.0)
    assert report.mean_brier_parsed < 0.05
    assert report.mean_crps_parsed < 0.05
    # ECE should be small: confidence high (~0.9), accuracy 1.0 → gap ~0.1.
    assert report.ece < 0.15


def test_confident_wrong_blows_up_ece() -> None:
    # Always predicts bin 1 with VERY_HIGH; true bin always 8.
    data = [
        CalibrationDatum(
            true_bin=SpeedupBin.EXTREME_SPEEDUP,
            estimate=_confident_wrong_estimate(SpeedupBin.SEVERE_SLOWDOWN),
        )
        for _ in range(20)
    ]
    report = evaluate_calibration(data)
    assert report.accuracy == pytest.approx(0.0)
    # Confidence near very_high (0.9), accuracy 0 → ECE near 0.9.
    assert report.ece > 0.6
    # CRPS in extreme-misprediction case should be near 1.
    assert report.mean_crps_parsed > 0.7


def test_unparsed_rows_contribute_to_full_brier_crps_only() -> None:
    data = [
        CalibrationDatum(
            true_bin=SpeedupBin.MINOR_SLOWDOWN,
            estimate=_confident_correct_estimate(SpeedupBin.MINOR_SLOWDOWN),
        ),
        CalibrationDatum(
            true_bin=SpeedupBin.MINOR_SPEEDUP, estimate=None
        ),
    ]
    report = evaluate_calibration(data)
    assert report.n_total == 2
    assert report.n_parsed == 1
    assert report.parsed_rate == pytest.approx(0.5)
    # parsed-only is correct → small; full includes the uniform fallback → larger.
    assert report.mean_brier_parsed < report.mean_brier
    assert report.mean_crps_parsed < report.mean_crps


def test_empty_data_is_a_zero_report() -> None:
    report = evaluate_calibration([])
    assert report.n_total == 0
    assert report.n_parsed == 0
    assert report.accuracy == pytest.approx(0.0)
    assert report.ece == pytest.approx(0.0)
    assert report.likert_mapping == LikertNumericMapping()


def test_reliability_bins_align_with_n_buckets() -> None:
    data = [
        CalibrationDatum(
            true_bin=SpeedupBin.MINOR_SLOWDOWN,
            estimate=_confident_correct_estimate(SpeedupBin.MINOR_SLOWDOWN),
        )
    ]
    report = evaluate_calibration(data, n_reliability_buckets=10)
    assert len(report.reliability_bins) == 10
    populated = [b for b in report.reliability_bins if b.count > 0]
    assert len(populated) == 1
    assert populated[0].accuracy == pytest.approx(1.0)
