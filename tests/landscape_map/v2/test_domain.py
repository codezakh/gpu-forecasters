"""Domain tests for v2: bin formula, simplex invariants, renormalization."""

from __future__ import annotations

import math

import pytest

from gpu_forecasters.landscape_map.v2.domain import (
    SUCCESS_BINS,
    KernelRuntimeEstimate,
    SpeedupBin,
    renormalize,
)


# ---------------------------------------------------------------------------
# SpeedupBin.from_speedup
# ---------------------------------------------------------------------------


def test_from_speedup_severe_slowdown() -> None:
    assert SpeedupBin.from_speedup(0.10) == SpeedupBin.SEVERE_SLOWDOWN


def test_from_speedup_unity_lands_in_minor_slowdown() -> None:
    # Half-octave bin at S=1 lands in bin 4 (the comments use the
    # interval (0.707, 1.0] but the formula floor(2*log2(S))+4
    # actually places S=1 in bin 4 because floor(0)=0 -> 0+4=4. Lock
    # this in so callers know which bin S=1 maps to.
    assert SpeedupBin.from_speedup(1.0) == SpeedupBin.MINOR_SLOWDOWN


def test_from_speedup_small_speedup() -> None:
    # S = 1.5 -> 2*log2(1.5) = 1.17 -> floor = 1 -> bin 5
    assert SpeedupBin.from_speedup(1.5) == SpeedupBin.MINOR_SPEEDUP


def test_from_speedup_extreme_speedup_clamps() -> None:
    assert SpeedupBin.from_speedup(1000.0) == SpeedupBin.EXTREME_SPEEDUP


def test_from_speedup_rejects_nonpositive() -> None:
    with pytest.raises(ValueError):
        SpeedupBin.from_speedup(0.0)
    with pytest.raises(ValueError):
        SpeedupBin.from_speedup(-1.0)


# ---------------------------------------------------------------------------
# renormalize
# ---------------------------------------------------------------------------


def test_renormalize_preserves_shape_and_returns_total() -> None:
    raw = {b: 0.10 for b in SUCCESS_BINS}
    raw[SpeedupBin.MINOR_SPEEDUP] = 0.30
    distribution, total = renormalize(raw)
    assert math.isclose(total, sum(raw.values()))
    assert math.isclose(sum(distribution.values()), 1.0, abs_tol=1e-9)
    # Order preserved with respect to SUCCESS_BINS
    assert set(distribution.keys()) == set(SUCCESS_BINS)


def test_renormalize_keeps_relative_proportions() -> None:
    raw = {b: 0.0 for b in SUCCESS_BINS}
    raw[SpeedupBin.MINOR_SPEEDUP] = 0.6
    raw[SpeedupBin.SIGNIFICANT_SPEEDUP] = 0.2
    distribution, _ = renormalize(raw)
    assert math.isclose(
        distribution[SpeedupBin.MINOR_SPEEDUP]
        / distribution[SpeedupBin.SIGNIFICANT_SPEEDUP],
        3.0,
    )


def test_renormalize_rejects_zero_sum() -> None:
    raw = {b: 0.0 for b in SUCCESS_BINS}
    with pytest.raises(ValueError):
        renormalize(raw)


# ---------------------------------------------------------------------------
# KernelRuntimeEstimate invariants
# ---------------------------------------------------------------------------


def _uniform_distribution() -> dict[SpeedupBin, float]:
    return {b: 1 / len(SUCCESS_BINS) for b in SUCCESS_BINS}


def test_estimate_accepts_well_formed_distribution() -> None:
    KernelRuntimeEstimate(
        predicted_bin=SpeedupBin.MINOR_SPEEDUP,
        bin_probabilities=_uniform_distribution(),
        reasoning="ok",
        raw_probability_sum=1.0,
    )


def test_estimate_rejects_failure_predicted_bin() -> None:
    with pytest.raises(ValueError):
        KernelRuntimeEstimate(
            predicted_bin=SpeedupBin.FAILURE,
            bin_probabilities=_uniform_distribution(),
            reasoning="ok",
            raw_probability_sum=1.0,
        )


def test_estimate_rejects_missing_bin() -> None:
    bp = _uniform_distribution()
    del bp[SpeedupBin.MINOR_SPEEDUP]
    with pytest.raises(ValueError):
        KernelRuntimeEstimate(
            predicted_bin=SpeedupBin.MINOR_SPEEDUP,
            bin_probabilities=bp,
            reasoning="ok",
            raw_probability_sum=1.0,
        )


def test_estimate_rejects_unrenormalized_distribution() -> None:
    bp = {b: 0.05 for b in SUCCESS_BINS}  # sums to 0.4, not 1
    with pytest.raises(ValueError):
        KernelRuntimeEstimate(
            predicted_bin=SpeedupBin.MINOR_SPEEDUP,
            bin_probabilities=bp,
            reasoning="ok",
            raw_probability_sum=0.4,
        )


def test_estimate_probability_of_failure_is_zero() -> None:
    est = KernelRuntimeEstimate(
        predicted_bin=SpeedupBin.MINOR_SPEEDUP,
        bin_probabilities=_uniform_distribution(),
        reasoning="ok",
        raw_probability_sum=1.0,
    )
    assert est.probability_of(SpeedupBin.FAILURE) == 0.0


def test_estimate_bin_probability_list_is_ordered() -> None:
    est = KernelRuntimeEstimate(
        predicted_bin=SpeedupBin.MINOR_SPEEDUP,
        bin_probabilities=_uniform_distribution(),
        reasoning="ok",
        raw_probability_sum=1.0,
    )
    expected = [est.bin_probabilities[b] for b in SUCCESS_BINS]
    assert est.bin_probability_list() == expected
