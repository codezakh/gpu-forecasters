"""
Tests for SpeedupBin.from_speedup() binning algorithm.

The formula under test: i = floor(2 * log2(S)) + 4, clamped to [1, 8].
"""

import pytest

from gpu_forecasters.landscape_map.v1.domain import SpeedupBin


@pytest.mark.parametrize(
    "speedup, expected_bin",
    [
        # Values chosen to land squarely within each bin according to the formula.
        # S=0.4: floor(2*log2(0.4))+4 = floor(-2.644)+4 = -3+4 = 1
        (0.4, SpeedupBin.SEVERE_SLOWDOWN),
        # S=0.6: floor(2*log2(0.6))+4 = floor(-1.474)+4 = -2+4 = 2
        (0.6, SpeedupBin.SIGNIFICANT_SLOWDOWN),
        # S=0.8: floor(2*log2(0.8))+4 = floor(-0.644)+4 = -1+4 = 3
        (0.8, SpeedupBin.MODERATE_SLOWDOWN),
        # S=1.0: floor(2*log2(1.0))+4 = floor(0)+4 = 4
        (1.0, SpeedupBin.MINOR_SLOWDOWN),
        # S=1.5: floor(2*log2(1.5))+4 = floor(1.17)+4 = 1+4 = 5
        (1.5, SpeedupBin.MINOR_SPEEDUP),
        # S=2.0: floor(2*log2(2.0))+4 = floor(2)+4 = 2+4 = 6
        (2.0, SpeedupBin.SIGNIFICANT_SPEEDUP),
        # S=3.0: floor(2*log2(3.0))+4 = floor(3.17)+4 = 3+4 = 7
        (3.0, SpeedupBin.HIGH_SPEEDUP),
        # S=5.0: floor(2*log2(5.0))+4 = floor(4.64)+4 = 4+4 = 8
        (5.0, SpeedupBin.EXTREME_SPEEDUP),
    ],
)
def test_speedup_bin_from_speedup_formula(speedup: float, expected_bin: SpeedupBin) -> None:
    assert SpeedupBin.from_speedup(speedup) == expected_bin


def test_speedup_bin_clamping() -> None:
    # Very small speedup: raw index = floor(2*log2(0.001))+4 = floor(-19.93)+4 = -16,
    # which is far below 1 and must be clamped to bin 1.
    assert SpeedupBin.from_speedup(0.001) == SpeedupBin.SEVERE_SLOWDOWN

    # Very large speedup: raw index = floor(2*log2(10000))+4 = floor(26.57)+4 = 30,
    # which is far above 8 and must be clamped to bin 8.
    assert SpeedupBin.from_speedup(10000.0) == SpeedupBin.EXTREME_SPEEDUP


def test_speedup_bin_raises_on_non_positive() -> None:
    # The guard clause `if speedup <= 0: raise ValueError` must trigger for zero
    # and negative values — log2 would be undefined or imaginary.
    with pytest.raises(ValueError):
        SpeedupBin.from_speedup(0.0)
    with pytest.raises(ValueError):
        SpeedupBin.from_speedup(-1.0)
