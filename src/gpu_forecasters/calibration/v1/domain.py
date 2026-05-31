"""Types for calibration evaluation of speedup-bin surrogates.

A ``CalibrationDatum`` pairs a ground-truth bin with whatever the
surrogate emitted on that input. The estimate is ``None`` if the
rollout failed to parse; downstream evaluation handles parsed and
unparsed rows differently (see ``evaluator.py``).
"""

from __future__ import annotations

from pydantic import BaseModel

from gpu_forecasters.landscape_map.v1.domain import (
    KernelRuntimeEstimate,
    LikertConfidence,
    SpeedupBin,
)


# The eight ordered bins the surrogate predicts over. Failure bin is
# excluded — current trainers (e0117, e0121) drop failure-truth rows
# before training, and the verbalized distribution does not include a
# failure entry.
PREDICTED_BINS: tuple[SpeedupBin, ...] = (
    SpeedupBin.SEVERE_SLOWDOWN,
    SpeedupBin.SIGNIFICANT_SLOWDOWN,
    SpeedupBin.MODERATE_SLOWDOWN,
    SpeedupBin.MINOR_SLOWDOWN,
    SpeedupBin.MINOR_SPEEDUP,
    SpeedupBin.SIGNIFICANT_SPEEDUP,
    SpeedupBin.HIGH_SPEEDUP,
    SpeedupBin.EXTREME_SPEEDUP,
)


class LikertNumericMapping(BaseModel, frozen=True):
    """Maps the 5-point Likert scale to numeric prior probabilities.

    Defaults follow the values proposed in the ZAI-58 spec. Two
    properties matter for the downstream metrics:

    * The ordering ``very_low < low < moderate < high < very_high`` is
      preserved — without this the verbalized distribution would no
      longer be monotone in confidence.
    * The values are not required to sum to one across the five Likert
      levels; ``bin_distribution`` renormalizes after looking each bin
      up, so what matters is only the relative weight between levels.

    Different mappings shift Brier/CRPS values without changing
    rankings (the renormalized distribution depends only on the ratio
    of the looked-up values), but ECE and entropy are sensitive to
    absolute values. The mapping is therefore logged inside every
    ``CalibrationReport``.
    """

    very_low: float = 0.05
    low: float = 0.15
    moderate: float = 0.30
    high: float = 0.60
    very_high: float = 0.90

    def numeric_for(self, level: LikertConfidence) -> float:
        match level:
            case LikertConfidence.VERY_LOW:
                return self.very_low
            case LikertConfidence.LOW:
                return self.low
            case LikertConfidence.MODERATE:
                return self.moderate
            case LikertConfidence.HIGH:
                return self.high
            case LikertConfidence.VERY_HIGH:
                return self.very_high


class CalibrationDatum(BaseModel, frozen=True):
    """One held-out evaluation row.

    ``estimate`` is ``None`` when the surrogate failed to produce a
    parseable response. The evaluator counts these toward
    ``parsed_rate`` and applies a uniform-fallback distribution for
    proper-scoring-rule totals; accuracy and ECE skip them since
    ``predicted_bin`` is undefined.
    """

    true_bin: SpeedupBin
    estimate: KernelRuntimeEstimate | None


class ReliabilityBin(BaseModel, frozen=True):
    """One row of a reliability diagram.

    ``confidence_low`` and ``confidence_high`` are the bucket edges
    over the predicted bin's verbalized confidence (numeric, after
    ``LikertNumericMapping``). ``mean_confidence`` is the average of
    that confidence within the bucket; ``accuracy`` is the empirical
    fraction of rows in the bucket where ``predicted_bin == true_bin``.
    A perfectly calibrated model has ``mean_confidence ≈ accuracy`` in
    every populated bucket.
    """

    confidence_low: float
    confidence_high: float
    mean_confidence: float
    accuracy: float
    count: int


class CalibrationReport(BaseModel, frozen=True):
    """Aggregate calibration metrics for one surrogate over a held-out set.

    All scalar metrics except ``parsed_rate`` are computed over the
    parsed subset, with the exception of ``mean_brier`` and
    ``mean_crps`` which include unparsed rows scored against a uniform
    fallback distribution (so a model that fails to parse cannot dodge
    its scoring-rule penalty by simply not answering). The two
    quantities are reported separately so the contribution of failures
    to the headline score is visible:

    * ``mean_brier_parsed`` / ``mean_crps_parsed`` — over parsed rows
      only.
    * ``mean_brier`` / ``mean_crps`` — over the full set including the
      uniform fallback for unparsed rows.

    ``mean_entropy`` is the average Shannon entropy (in nats) of the
    renormalized 8-bin distribution over parsed rows. Reading it
    alongside ``accuracy`` distinguishes a calibrated-because-spread
    model from a calibrated-because-knowledgeable one.
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
    reliability_bins: list[ReliabilityBin]
    likert_mapping: LikertNumericMapping
