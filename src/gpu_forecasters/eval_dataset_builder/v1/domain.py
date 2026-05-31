"""Domain values, protocols, and the speedup-band accessor.

The canonical on-disk artifact is a JSONL of ``KernelRuntimeComparison``.
Everything else here exists to produce that artifact: the desired set
shape (``NumKernelsForSpeedupBin``), the request handed to a kernel
generator when harvest leaves a bin under-populated
(``RequestForKernelInGoalSpeedupBin``), the cached record of one
(generate, evaluate) cycle (``KernelGenerationAttempt``), the audit
record shipped alongside the JSONL (``EvalSetManifest``), the
filler's request/result/spec types, the run-summary record, and the
``HarvestedKernelSource`` protocol the orchestrator depends on.

``SpeedupBand`` and ``speedup_band_for_bin`` translate a ``SpeedupBin``
into the concrete ``[lo, hi)`` interval, log-axis midpoint, and
prompt-friendly display string used by the goal-conditioned search.

The evaluator side is *not* a new protocol — the bin filler depends
directly on ``GpuModeKernelModalProvider``, so we don't reinvent a
parallel surface for what is already a well-defined seam.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path
from typing import Generic, Literal, Protocol, TypeAlias

from pydantic import BaseModel, ConfigDict, model_validator

from gpu_forecasters.gpu_mode_kernel.aggregation import AggregationMethod
from gpu_forecasters.gpu_mode_kernel.core import CaseSpeedupT, GpuModeKernelObservation
from gpu_forecasters.hill_climbing.domain import Evaluation
from gpu_forecasters.landscape_map.v1.domain import HardwareContext, SpeedupBin


# ---------------------------------------------------------------------------
# Speedup band — geometry of one ``SpeedupBin``.
# ---------------------------------------------------------------------------


class SpeedupBand(BaseModel, frozen=True):
    """The half-open speedup interval ``[lo, hi)`` corresponding to a
    ``SpeedupBin``, plus the log-axis midpoint and a prompt-friendly
    display string.

    ``hi`` is ``math.inf`` for the unbounded top bin. ``midpoint`` is
    the geometric mean of ``lo`` and ``hi`` — the natural centerpoint
    on a log-speedup axis — and is the target the goal-conditioned
    search aims for.
    """

    lo: float
    hi: float
    midpoint: float
    display: str


def speedup_band_for_bin(bin: SpeedupBin) -> SpeedupBand:
    """Return the ``SpeedupBand`` for ``bin``.

    Bin formula in ``SpeedupBin.from_speedup`` is
    ``i = clamp(floor(2 * log2(S)) + 4, 1, 8)``, so for the unclamped
    bins (2..7) the speedup interval is ``[2^((i-4)/2), 2^((i-3)/2))``.
    Bins 1 and 8 are clamp targets; we pick a sensible finite half-octave
    window for the midpoint computation while keeping the display string
    reflective of the open clamp edge.

    Raises ``ValueError`` on ``SpeedupBin.FAILURE`` — failure is not a
    speedup the search can aim for.
    """
    if bin is SpeedupBin.FAILURE:
        raise ValueError(
            "SpeedupBin.FAILURE has no speedup band — failure is not a "
            "speedup the search can aim for."
        )
    i = int(bin)
    if 2 <= i <= 7:
        lo = 2.0 ** ((i - 4) / 2.0)
        hi = 2.0 ** ((i - 3) / 2.0)
        mid = math.sqrt(lo * hi)
        return SpeedupBand(lo=lo, hi=hi, midpoint=mid, display=f"{lo:.2f}×–{hi:.2f}×")
    if i == 1:
        # Clamp target — anything ≤ 0.5×. Use the half-octave window
        # [0.25×, 0.5×) for the midpoint; that matches the bin's label
        # while keeping a finite range. ``lo`` is reported as 0.0 because
        # the bin's true lower edge is unbounded.
        lo_window, hi = 0.25, 0.5
        mid = math.sqrt(lo_window * hi)
        return SpeedupBand(lo=0.0, hi=hi, midpoint=mid, display="≤ 0.50×")
    # i == 8 — anything > 4×. Use the half-octave window [4×, 8×) for
    # the midpoint; ``hi`` is reported as ``math.inf`` because the bin
    # is unbounded above.
    lo = 4.0
    hi_window = 8.0
    mid = math.sqrt(lo * hi_window)
    return SpeedupBand(lo=lo, hi=math.inf, midpoint=mid, display="> 4.00×")


# ---------------------------------------------------------------------------
# Eval-set values.
# ---------------------------------------------------------------------------


# Declarative spec of how many kernels we want per bin.
NumKernelsForSpeedupBin: TypeAlias = dict[SpeedupBin, int]


# The eval set itself: kernels grouped by their measured speedup bin.
EvalSet: TypeAlias = dict[SpeedupBin, list["KernelRuntimeComparison"]]


class KernelRuntimeComparison(BaseModel, frozen=True):
    """One row of the eval set — a candidate kernel measured against a reference.

    ``true_bin`` is logically derived from ``aggregated_speedup``; we store
    both for downstream convenience but enforce the invariant at construction
    time so the two cannot drift.
    """

    reference_code: str
    candidate_code: str
    hardware: HardwareContext
    aggregated_speedup: float
    true_bin: SpeedupBin
    source_id: str

    @model_validator(mode="after")
    def _bin_matches_speedup(self) -> "KernelRuntimeComparison":
        derived = SpeedupBin.from_speedup(self.aggregated_speedup)
        if derived != self.true_bin:
            raise ValueError(
                f"true_bin={self.true_bin!r} disagrees with bin derived from "
                f"aggregated_speedup={self.aggregated_speedup} -> {derived!r}"
            )
        return self


class RequestForKernelInGoalSpeedupBin(BaseModel, frozen=True):
    """A request for a kernel whose runtime falls in ``target_bin``.

    Carries everything a generator needs to produce a candidate aimed at the
    target: the bin window (via ``target_bin``), the reference to beat, and
    the hardware the resulting kernel will be measured on.
    """

    target_bin: SpeedupBin
    reference_code: str
    hardware: HardwareContext


class KernelGenerationAttempt(BaseModel, Generic[CaseSpeedupT], frozen=True):
    """The cached record of one (generate, evaluate) cycle.

    Generic over the pack's per-case speedup type because the carried
    ``Evaluation`` is parameterized by
    ``GpuModeKernelObservation[CaseSpeedupT]``. Stands alone — carries
    the request that produced it, so a cache reader can both decide
    acceptance and project to a ``KernelRuntimeComparison`` if accepted,
    without external lookup.
    """

    request: RequestForKernelInGoalSpeedupBin
    candidate_code: str
    evaluation: Evaluation[GpuModeKernelObservation[CaseSpeedupT]]


class EvalSetManifest(BaseModel, frozen=True):
    """Audit record shipped alongside the eval-set JSONL.

    Provenance: which search produced the harvested portion, how
    harvest vs. generation contributed per bin, how many generation
    attempts each bin took, the hardware the whole set was measured
    on, and a generation timestamp.
    """

    source_search_tag: str
    hardware: HardwareContext
    harvested_per_bin: dict[SpeedupBin, int]
    generated_per_bin: dict[SpeedupBin, int]
    attempts_per_bin: dict[SpeedupBin, int]
    generated_at: datetime


class EvalDataset(BaseModel, frozen=True):
    """A loaded eval dataset: rows + the manifest produced alongside them.

    The reciprocal of ``write_eval_set``: anything that function writes
    can be read back into one of these via ``read_eval_dataset``. The
    two halves of the on-disk artifact are bundled here so callers that
    want both don't have to assemble them.
    """

    comparisons: list[KernelRuntimeComparison]
    manifest: EvalSetManifest

    def by_bin(self) -> EvalSet:
        out: dict[SpeedupBin, list[KernelRuntimeComparison]] = {}
        for row in self.comparisons:
            out.setdefault(row.true_bin, []).append(row)
        return out


# ---------------------------------------------------------------------------
# Harvest seam.
# ---------------------------------------------------------------------------


class HarvestedKernelSource(Protocol):
    """Caller-provided source of kernels harvested from a prior search.

    Implementors adapt whatever checkpoint or log format the source
    search produced into ``KernelRuntimeComparison`` rows the eval-set
    builder consumes. The library defines no checkpoint shape of its
    own; harvest is the caller's responsibility.
    """

    def __call__(self) -> Iterable[KernelRuntimeComparison]: ...


# ---------------------------------------------------------------------------
# Filler request/result/specs.
# ---------------------------------------------------------------------------


SeedSelectionStrategy = Literal["closest_to_midpoint"]


class MutationProviderSpec(BaseModel):
    """LLM-mutation provider knobs. Stable across bins for one pipeline run."""

    model_config = ConfigDict(frozen=True)

    model_slug: str
    gpu_name: str
    triton_version: str = "3.3.1"
    max_llm_concurrency: int = 8
    num_retries: int = 4
    request_timeout_s: float = 600.0
    temperature: float = 1.0
    max_tokens: int | None = None


class EvaluationProviderSpec(BaseModel):
    """Modal evaluation provider knobs. Stable across bins for one pipeline run."""

    model_config = ConfigDict(frozen=True)

    eval_gpu: str = "A100-80GB"
    aggregator: AggregationMethod = "geomean"
    max_in_flight: int = 10
    max_containers: int = 10
    get_timeout_s: float = 1200.0


class BinFillRequest(BaseModel):
    """Per-call input to ``BinFiller.fill``."""

    model_config = ConfigDict(frozen=True)

    target_bin: SpeedupBin
    harvested: list[KernelRuntimeComparison]
    output_dir: Path
    seed_selection_strategy: SeedSelectionStrategy = "closest_to_midpoint"


class BinFillResult(BaseModel):
    """Per-call output from ``BinFiller.fill``."""

    model_config = ConfigDict(frozen=True)

    target_bin: SpeedupBin
    in_target_kernels: list[KernelRuntimeComparison]
    events_log_path: Path
    summary: "RunSummary"


# ---------------------------------------------------------------------------
# Run summary.
# ---------------------------------------------------------------------------


class RunSummary(BaseModel):
    """Persisted post-run summary for one ``BinFiller.fill`` invocation."""

    model_config = ConfigDict(frozen=True)

    target_bin: str
    target_band_lo: float
    target_band_hi: float
    target_midpoint_speedup: float

    seed_source_id: str | None
    seed_speedup_at_harvest: float | None
    seed_speedup_after_bootstrap_eval: float | None

    model_slug: str
    search_config: dict[str, object]

    total_candidates_evaluated: int
    per_bin_count_all_candidates: dict[str, int]
    per_bin_count_archive_at_end: dict[str, int]
    in_target_bin_count_all_candidates: int
    in_target_bin_count_archive_at_end: int

    wall_clock_seconds: float
