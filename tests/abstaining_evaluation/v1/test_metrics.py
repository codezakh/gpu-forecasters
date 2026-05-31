"""Tests for risk-coverage sweep, AURC, selective@coverage, matched-coverage helper."""

from __future__ import annotations

import math

import pytest

from gpu_forecasters.abstaining_evaluation.v1 import (
    AbstainDecision,
    BinaryMismatchRisk,
    MaxProbScore,
    Predict,
    PredictOrAbstain,
    RiskCoveragePoint,
    aurc,
    decision_set_agreement,
    match_coverage_with_threshold,
    risk_coverage_curve,
    selective_at_coverage,
)
from gpu_forecasters.eval_dataset_builder.v1 import KernelRuntimeComparison
from gpu_forecasters.landscape_map.v2 import KernelRuntimeEstimate
from gpu_forecasters.landscape_map.v2 import SpeedupBin

from ._fixtures import make_comparison, make_estimate, true_bin_v2


def _build_dataset() -> tuple[
    list[KernelRuntimeEstimate | None], list[KernelRuntimeComparison]
]:
    """One correct high-conf row + one wrong low-conf row + one parse failure."""
    comparisons = [
        make_comparison(source_id="a", speedup=1.5),
        make_comparison(source_id="b", speedup=1.5),
        make_comparison(source_id="c", speedup=1.5),
    ]
    correct_bin = true_bin_v2(comparisons[0])
    wrong_bin = SpeedupBin.EXTREME_SPEEDUP
    assert wrong_bin != correct_bin
    estimates = [
        make_estimate(predicted_bin=correct_bin, confidence=0.9),
        make_estimate(predicted_bin=wrong_bin, confidence=0.2),
        None,  # parse failure
    ]
    return estimates, comparisons


def test_risk_coverage_curve_monotone_in_threshold() -> None:
    estimates, comparisons = _build_dataset()
    curve = risk_coverage_curve(
        estimates=estimates,
        comparisons=comparisons,
        score=MaxProbScore(),
        risk=BinaryMismatchRisk(),
    )
    # Coverage must be monotone non-increasing as threshold rises.
    # ``threshold`` is ``float | None`` on the dataclass; the sweep
    # always sets it, so cast to assist the type checker.
    by_thr = sorted(curve, key=lambda p: float("inf") if p.threshold is None else p.threshold)
    for a, b in zip(by_thr[:-1], by_thr[1:]):
        assert b.coverage <= a.coverage + 1e-12


def test_risk_coverage_curve_max_coverage_bounded_by_parsed_rate() -> None:
    estimates, comparisons = _build_dataset()
    curve = risk_coverage_curve(
        estimates=estimates,
        comparisons=comparisons,
        score=MaxProbScore(),
        risk=BinaryMismatchRisk(),
    )
    max_coverage = max(p.coverage for p in curve)
    # Two of three rows are parseable.
    assert max_coverage == pytest.approx(2.0 / 3.0, abs=1e-12)


def test_risk_coverage_high_conf_only_yields_zero_binary_risk() -> None:
    estimates, comparisons = _build_dataset()
    curve = risk_coverage_curve(
        estimates=estimates,
        comparisons=comparisons,
        score=MaxProbScore(),
        risk=BinaryMismatchRisk(),
    )
    # At τ between 0.2 and 0.9, only the correct row predicts.
    point = next(p for p in curve if p.coverage == pytest.approx(1.0 / 3.0))
    assert point.risk == pytest.approx(0.0)


def test_aurc_drops_nan_points() -> None:
    estimates, comparisons = _build_dataset()
    curve = risk_coverage_curve(
        estimates=estimates,
        comparisons=comparisons,
        score=MaxProbScore(),
        risk=BinaryMismatchRisk(),
    )
    value = aurc(curve)
    assert not math.isnan(value)
    assert value >= 0.0


def test_selective_at_coverage_interpolates() -> None:
    estimates, comparisons = _build_dataset()
    curve = risk_coverage_curve(
        estimates=estimates,
        comparisons=comparisons,
        score=MaxProbScore(),
        risk=BinaryMismatchRisk(),
    )
    # At max coverage (2/3) we should have one wrong out of two = 0.5.
    assert selective_at_coverage(curve, 2.0 / 3.0) == pytest.approx(0.5)
    # At 1/3 coverage we should have only the correct row → 0.
    assert selective_at_coverage(curve, 1.0 / 3.0) == pytest.approx(0.0)


def test_match_coverage_threshold_yields_target_coverage() -> None:
    estimates, _ = _build_dataset()
    policy = match_coverage_with_threshold(
        estimates=estimates,
        target_coverage=1.0 / 3.0,
        score=MaxProbScore(),
    )
    decisions = [policy(e) for e in estimates]
    n_predicted = sum(1 for d in decisions if isinstance(d, Predict))
    assert n_predicted == 1


def test_decision_set_agreement_jaccard() -> None:
    a = [AbstainDecision(), AbstainDecision(), Predict(estimate=make_estimate(
        predicted_bin=SpeedupBin.MINOR_SPEEDUP, confidence=0.5
    ))]
    b = [AbstainDecision(), Predict(estimate=make_estimate(
        predicted_bin=SpeedupBin.MINOR_SPEEDUP, confidence=0.5
    )), Predict(estimate=make_estimate(
        predicted_bin=SpeedupBin.MINOR_SPEEDUP, confidence=0.5
    ))]
    # a abstains on {0,1}; b abstains on {0}; ∩={0}, ∪={0,1} → 0.5.
    assert decision_set_agreement(a, b) == pytest.approx(0.5)
