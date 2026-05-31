"""Tests for log-distance-minimizing seed selection."""

from __future__ import annotations

import math

import pytest

from gpu_forecasters.eval_dataset_builder.v1.domain import (
    KernelRuntimeComparison,
    speedup_band_for_bin,
)
from gpu_forecasters.eval_dataset_builder.v1.seed_selection import (
    select_seed,
    select_seed_kernel_closest_to_target_midpoint,
)
from gpu_forecasters.landscape_map.v1.domain import HardwareContext, SpeedupBin


_HARDWARE = HardwareContext(
    device_name="fake-gpu",
    compute_capability=(0, 0),
    total_global_memory_gb=0.0,
    multiprocessor_count=0,
    max_threads_per_multiprocessor=0,
    clock_rate_ghz=0.0,
    memory_clock_rate_ghz=0.0,
    memory_bus_width_bits=0,
)


def _comparison(*, source_id: str, speedup: float) -> KernelRuntimeComparison:
    return KernelRuntimeComparison(
        reference_code="REF",
        candidate_code=f"CAND[{source_id}]",
        hardware=_HARDWARE,
        aggregated_speedup=speedup,
        true_bin=SpeedupBin.from_speedup(speedup),
        source_id=source_id,
    )


def test_picks_closest_in_log_space() -> None:
    bin_ = SpeedupBin.HIGH_SPEEDUP
    mid = speedup_band_for_bin(bin_).midpoint  # ≈ 2.83
    closest = _comparison(source_id="b", speedup=mid)
    candidates = [
        _comparison(source_id="a", speedup=mid * 2),
        closest,
        _comparison(source_id="c", speedup=mid / 2),
    ]
    pick = select_seed_kernel_closest_to_target_midpoint(candidates, target_bin=bin_)
    assert pick.source_id == "b"


def test_picks_closest_when_above_and_below() -> None:
    bin_ = SpeedupBin.HIGH_SPEEDUP
    mid = speedup_band_for_bin(bin_).midpoint
    above = mid * 1.3
    below = mid / 1.31  # slightly farther in log space
    candidates = [
        _comparison(source_id="above", speedup=above),
        _comparison(source_id="below", speedup=below),
    ]
    pick = select_seed_kernel_closest_to_target_midpoint(candidates, target_bin=bin_)
    expected = (
        "above"
        if abs(math.log(above) - math.log(mid))
        < abs(math.log(below) - math.log(mid))
        else "below"
    )
    assert pick.source_id == expected


def test_empty_input_raises() -> None:
    with pytest.raises(ValueError, match="no harvested kernels"):
        _ = select_seed_kernel_closest_to_target_midpoint(
            [], target_bin=SpeedupBin.HIGH_SPEEDUP
        )


def test_select_seed_uses_harvest_when_available() -> None:
    from gpu_forecasters.gpu_mode_kernel.packs.trimul import TRIMUL_PACK

    bin_ = SpeedupBin.HIGH_SPEEDUP
    mid = speedup_band_for_bin(bin_).midpoint
    closest = _comparison(source_id="b", speedup=mid)
    candidates = [
        _comparison(source_id="a", speedup=mid * 2),
        closest,
    ]
    seed = select_seed(candidates, target_bin=bin_, pack=TRIMUL_PACK)
    assert seed.source is not None
    assert seed.source.source_id == "b"
    assert seed.program_code == closest.candidate_code


def test_select_seed_falls_back_to_pack_seed_kernel_code() -> None:
    from gpu_forecasters.gpu_mode_kernel.packs.trimul import TRIMUL_PACK

    seed = select_seed([], target_bin=SpeedupBin.HIGH_SPEEDUP, pack=TRIMUL_PACK)
    assert seed.source is None
    assert seed.program_code == TRIMUL_PACK.seed_kernel_code
