"""Tiny fixtures for the abstaining-evaluation tests.

We avoid the ``KernelRuntimeComparison`` validator that requires
``true_bin == SpeedupBin.from_speedup(aggregated_speedup)`` by
constructing values that satisfy it.
"""

from __future__ import annotations

from arid_badger.eval_dataset_builder.v1 import KernelRuntimeComparison
from arid_badger.eval_dataset_builder.v1.domain import HardwareContext
from arid_badger.landscape_map.v1.domain import SpeedupBin as SpeedupBinV1
from arid_badger.landscape_map.v2 import (
    SUCCESS_BINS,
    KernelRuntimeEstimate,
    SpeedupBin,
)


_HW = HardwareContext(
    device_name="A100-80GB",
    compute_capability=(8, 0),
    total_global_memory_gb=80.0,
    multiprocessor_count=108,
    max_threads_per_multiprocessor=2048,
    clock_rate_ghz=1.41,
    memory_clock_rate_ghz=1.215,
    memory_bus_width_bits=5120,
)


def make_comparison(
    *, source_id: str, speedup: float
) -> KernelRuntimeComparison:
    return KernelRuntimeComparison(
        reference_code="def reference(): pass",
        candidate_code=f"# {source_id}",
        hardware=_HW,
        aggregated_speedup=speedup,
        true_bin=SpeedupBinV1.from_speedup(speedup),
        source_id=source_id,
    )


def true_bin_v2(comparison: KernelRuntimeComparison) -> SpeedupBin:
    """Convert ``KernelRuntimeComparison.true_bin`` (v1 SpeedupBin) to v2.

    The two enums share underlying ints. v2's library code accepts only
    v2 ``SpeedupBin``; tests round-trip through the underlying int so
    the type checker stays happy.
    """
    return SpeedupBin(int(comparison.true_bin))


def make_estimate(
    *,
    predicted_bin: SpeedupBin,
    confidence: float,
) -> KernelRuntimeEstimate:
    """Build an estimate with ``confidence`` mass on ``predicted_bin``
    and the remainder split uniformly across the other seven bins.
    """
    if not 1.0 / 8.0 <= confidence <= 1.0:
        raise ValueError(
            f"confidence must be in [1/8, 1] (got {confidence!r})"
        )
    other = (1.0 - confidence) / (len(SUCCESS_BINS) - 1)
    probs: dict[SpeedupBin, float] = {b: other for b in SUCCESS_BINS}
    probs[predicted_bin] = confidence
    return KernelRuntimeEstimate(
        predicted_bin=predicted_bin,
        bin_probabilities=probs,
        reasoning="fixture",
        raw_probability_sum=1.0,
    )
