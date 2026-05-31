"""Invariants of ``speedup_band_for_bin``.

The speedup band geometry is load-bearing: the goal-conditioned eval
wrapper rewrites reward via ``log(midpoint)``, the seed picker scores
by distance to ``log(midpoint)``, and the prompt template renders
``display`` and ``[lo, hi)``. If any of these drift relative to
``SpeedupBin.from_speedup`` the search aims at the wrong target.
"""

from __future__ import annotations

import math

import pytest

from gpu_forecasters.eval_dataset_builder.v1.domain import (
    SpeedupBand,
    speedup_band_for_bin,
)
from gpu_forecasters.landscape_map.v1.domain import SpeedupBin


def test_failure_bin_raises() -> None:
    with pytest.raises(ValueError):
        _ = speedup_band_for_bin(SpeedupBin.FAILURE)


@pytest.mark.parametrize(
    "bin_, expected_lo, expected_hi",
    [
        (SpeedupBin.SIGNIFICANT_SLOWDOWN, 2.0 ** (-1.0), 2.0 ** (-0.5)),  # i=2
        (SpeedupBin.MODERATE_SLOWDOWN, 2.0 ** (-0.5), 2.0**0.0),  # i=3
        (SpeedupBin.MINOR_SLOWDOWN, 2.0**0.0, 2.0**0.5),  # i=4
        (SpeedupBin.MINOR_SPEEDUP, 2.0**0.5, 2.0**1.0),  # i=5
        (SpeedupBin.SIGNIFICANT_SPEEDUP, 2.0**1.0, 2.0**1.5),  # i=6
        (SpeedupBin.HIGH_SPEEDUP, 2.0**1.5, 2.0**2.0),  # i=7
    ],
)
def test_unclamped_bins_match_bin_formula(
    bin_: SpeedupBin, expected_lo: float, expected_hi: float
) -> None:
    band = speedup_band_for_bin(bin_)
    assert band.lo == pytest.approx(expected_lo)
    assert band.hi == pytest.approx(expected_hi)
    assert band.midpoint == pytest.approx(math.sqrt(expected_lo * expected_hi))


def test_severe_slowdown_clamp() -> None:
    band = speedup_band_for_bin(SpeedupBin.SEVERE_SLOWDOWN)
    # Lower edge is unbounded — reported as 0.0 — but the midpoint window
    # is [0.25, 0.5).
    assert band.lo == 0.0
    assert band.hi == pytest.approx(0.5)
    assert band.midpoint == pytest.approx(math.sqrt(0.25 * 0.5))
    assert "0.50×" in band.display


def test_extreme_speedup_clamp() -> None:
    band = speedup_band_for_bin(SpeedupBin.EXTREME_SPEEDUP)
    assert band.lo == pytest.approx(4.0)
    assert math.isinf(band.hi)
    assert band.midpoint == pytest.approx(math.sqrt(4.0 * 8.0))
    assert "4.00×" in band.display


def test_band_midpoint_round_trips_through_from_speedup() -> None:
    """The geometric-mean midpoint of every unclamped band must classify
    back into the same bin under ``SpeedupBin.from_speedup`` — otherwise
    the goal-conditioned reward function aims at a midpoint that lives
    in a neighboring bin.
    """
    for bin_ in (
        SpeedupBin.SIGNIFICANT_SLOWDOWN,
        SpeedupBin.MODERATE_SLOWDOWN,
        SpeedupBin.MINOR_SLOWDOWN,
        SpeedupBin.MINOR_SPEEDUP,
        SpeedupBin.SIGNIFICANT_SPEEDUP,
        SpeedupBin.HIGH_SPEEDUP,
    ):
        band = speedup_band_for_bin(bin_)
        assert SpeedupBin.from_speedup(band.midpoint) is bin_


def test_returns_speedup_band_type() -> None:
    band = speedup_band_for_bin(SpeedupBin.MINOR_SPEEDUP)
    assert isinstance(band, SpeedupBand)
