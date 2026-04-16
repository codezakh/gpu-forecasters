"""Unit tests for the speedup aggregation helper."""

from __future__ import annotations

import math

from arid_badger.hill_climbing.scoring_providers.trimul_modal import (
    _aggregate_speedups,
)


def test_geomean_basic() -> None:
    # geomean of [2.0, 8.0] = sqrt(16) = 4.0
    result = _aggregate_speedups([2.0, 8.0], "geomean")
    assert math.isclose(result, 4.0, rel_tol=1e-9)


def test_geomean_single() -> None:
    result = _aggregate_speedups([3.5], "geomean")
    assert math.isclose(result, 3.5, rel_tol=1e-9)


def test_geomean_uniform() -> None:
    result = _aggregate_speedups([5.0, 5.0, 5.0], "geomean")
    assert math.isclose(result, 5.0, rel_tol=1e-9)


def test_min() -> None:
    result = _aggregate_speedups([1.5, 3.0, 0.8], "min")
    assert result == 0.8


def test_min_single() -> None:
    result = _aggregate_speedups([2.0], "min")
    assert result == 2.0


def test_arith_mean() -> None:
    result = _aggregate_speedups([1.0, 2.0, 3.0], "arith_mean")
    assert math.isclose(result, 2.0, rel_tol=1e-9)


def test_arith_mean_single() -> None:
    result = _aggregate_speedups([4.0], "arith_mean")
    assert math.isclose(result, 4.0, rel_tol=1e-9)


def test_geomean_less_than_or_equal_arith_mean() -> None:
    """AM-GM inequality: geomean ≤ arith mean for non-negative values."""
    speedups = [1.2, 2.5, 0.9, 3.1]
    geo = _aggregate_speedups(speedups, "geomean")
    arith = _aggregate_speedups(speedups, "arith_mean")
    assert geo <= arith + 1e-12
