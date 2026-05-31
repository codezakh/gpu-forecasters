"""StubEstimator tests."""

from __future__ import annotations

import math

import pytest

from gpu_forecasters.landscape_map.v2.domain import (
    KernelImplementation,
    KernelRuntimeQuery,
    KernelTaskInfo,
    SpeedupBin,
)
from gpu_forecasters.landscape_map.v2.stub_estimator import StubEstimator


def _query() -> KernelRuntimeQuery:
    return KernelRuntimeQuery(
        task=KernelTaskInfo(op_name="vector_add", level_id=1, task_id=1),
        reference=KernelImplementation(
            kernel_name="ref", code="pass", runtime_ms=1.0
        ),
        candidate=KernelImplementation(
            kernel_name="cand", code="pass", runtime_ms=1.0
        ),
    )


def test_stub_default_concentrates_on_minor_slowdown() -> None:
    estimator = StubEstimator()
    estimate, usage = estimator.estimate(_query())
    assert estimate.predicted_bin == SpeedupBin.MINOR_SLOWDOWN
    assert estimate.bin_probabilities[SpeedupBin.MINOR_SLOWDOWN] > 0.5
    assert math.isclose(sum(estimate.bin_probabilities.values()), 1.0, abs_tol=1e-9)
    assert usage is None


def test_stub_custom_bin() -> None:
    estimator = StubEstimator(fixed_bin=SpeedupBin.EXTREME_SPEEDUP)
    estimate, _ = estimator.estimate(_query())
    assert estimate.predicted_bin == SpeedupBin.EXTREME_SPEEDUP
    # The fixed bin holds the most mass.
    sorted_by_mass = sorted(
        estimate.bin_probabilities.items(), key=lambda kv: kv[1], reverse=True
    )
    assert sorted_by_mass[0][0] == SpeedupBin.EXTREME_SPEEDUP


def test_stub_concentrated_mass_controls_distribution_shape() -> None:
    estimator = StubEstimator(
        fixed_bin=SpeedupBin.MINOR_SPEEDUP, concentrated_mass=0.5
    )
    estimate, _ = estimator.estimate(_query())
    assert math.isclose(
        estimate.bin_probabilities[SpeedupBin.MINOR_SPEEDUP], 0.5, abs_tol=1e-9
    )
    # Remaining 0.5 mass should be split uniformly across 7 bins.
    spread = 0.5 / 7
    for b, p in estimate.bin_probabilities.items():
        if b == SpeedupBin.MINOR_SPEEDUP:
            continue
        assert math.isclose(p, spread, abs_tol=1e-9)


def test_stub_rejects_failure_bin() -> None:
    with pytest.raises(ValueError):
        StubEstimator(fixed_bin=SpeedupBin.FAILURE)


def test_stub_rejects_invalid_concentrated_mass() -> None:
    with pytest.raises(ValueError):
        StubEstimator(concentrated_mass=0.0)
    with pytest.raises(ValueError):
        StubEstimator(concentrated_mass=1.5)
