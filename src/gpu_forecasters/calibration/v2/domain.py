"""Domain types for v2 calibration scoring."""

from __future__ import annotations

from pydantic import BaseModel

from gpu_forecasters.landscape_map.v2 import KernelRuntimeEstimate, SpeedupBin


class CalibrationDatum(BaseModel, frozen=True):
    """One held-out row.

    ``estimate=None`` is the parse-failure path; the evaluator scores
    those against a uniform fallback distribution so a model can't
    dodge the proper-scoring penalty by failing to answer.
    """

    true_bin: SpeedupBin
    estimate: KernelRuntimeEstimate | None


class ReliabilityBin(BaseModel, frozen=True):
    """One bucket of the winning-class reliability diagram."""

    confidence_low: float
    confidence_high: float
    mean_confidence: float
    accuracy: float
    count: int


class CalibrationReport(BaseModel, frozen=True):
    """Aggregate v2 calibration metrics over a held-out set.

    Field set mirrors :class:`gpu_forecasters.calibration.v1.CalibrationReport`
    so a single rendering pipeline can serve both. The only field
    without a v2 analogue is ``likert_mapping`` — v2 has no projection
    parameter — so it's omitted, and a ``mean_raw_probability_sum``
    field is added as a v2-specific calibration-health signal.
    """

    n_total: int
    n_parsed: int
    parsed_rate: float
    accuracy: float
    mean_entropy: float
    ece: float
    mean_brier: float
    mean_brier_parsed: float
    mean_crps: float
    mean_crps_parsed: float
    mean_nll: float
    mean_nll_parsed: float
    mean_raw_probability_sum: float | None
    reliability_bins: list[ReliabilityBin]
