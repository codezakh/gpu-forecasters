"""Pick a seed kernel for a goal-conditioned PUCT search.

Two functions:

* ``select_seed_kernel_closest_to_target_midpoint`` — pure log-distance
  minimization over harvested kernels. Raises if input is empty.
* ``select_seed`` — wraps the above with a cold-start fallback to
  ``pack.seed_kernel_code`` when harvest is empty. This is what the
  ``BinFiller`` calls; the pure helper stays available for inspection
  and tests.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from gpu_forecasters.gpu_mode_kernel.kernel_pack import KernelPack
from gpu_forecasters.landscape_map.v1.domain import SpeedupBin

from .domain import KernelRuntimeComparison, speedup_band_for_bin


@dataclass(frozen=True)
class SelectedSeed:
    """The seed chosen for a fill run.

    ``source`` is the harvest record when one was picked, or ``None``
    when the seed came from ``pack.seed_kernel_code`` (cold start).
    """

    program_code: str
    source: KernelRuntimeComparison | None


def select_seed_kernel_closest_to_target_midpoint(
    harvested_kernels: Sequence[KernelRuntimeComparison],
    *,
    target_bin: SpeedupBin,
) -> KernelRuntimeComparison:
    """Return the harvested kernel whose ``aggregated_speedup`` minimizes
    ``|log(s) - log(midpoint)|``. Ties are broken by ``source_id``
    lexicographic order.
    """
    if not harvested_kernels:
        raise ValueError("no harvested kernels to seed from")
    log_midpoint = math.log(speedup_band_for_bin(target_bin).midpoint)

    def sort_key(comparison: KernelRuntimeComparison) -> tuple[float, str]:
        speedup = comparison.aggregated_speedup
        if speedup <= 0:
            distance = math.inf
        else:
            distance = abs(math.log(speedup) - log_midpoint)
        return (distance, comparison.source_id)

    return min(harvested_kernels, key=sort_key)


def select_seed(
    harvested: Sequence[KernelRuntimeComparison],
    *,
    target_bin: SpeedupBin,
    pack: KernelPack[Any, Any],
) -> SelectedSeed:
    """Pick a seed for ``target_bin``. Falls back to ``pack.seed_kernel_code``
    when harvest is empty. Every ``KernelPack`` ships with a starter
    seed, so this guarantees a usable seed exists for any pack.
    """
    if harvested:
        comparison = select_seed_kernel_closest_to_target_midpoint(
            harvested, target_bin=target_bin
        )
        return SelectedSeed(program_code=comparison.candidate_code, source=comparison)
    return SelectedSeed(program_code=pack.seed_kernel_code, source=None)
