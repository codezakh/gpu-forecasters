"""Calibration metrics for ordinal speedup-bin surrogates.

See ``README.md`` for the design overview. Public surface re-exported
here so callers can write ``from gpu_forecasters.calibration.v1 import
evaluate_calibration, LikertNumericMapping``.
"""

from gpu_forecasters.calibration.v1.distribution import (
    bin_distribution,
    entropy,
    uniform_distribution,
)
from gpu_forecasters.calibration.v1.domain import (
    PREDICTED_BINS,
    CalibrationDatum,
    CalibrationReport,
    LikertNumericMapping,
    ReliabilityBin,
)
from gpu_forecasters.calibration.v1.ece import (
    expected_calibration_error,
    reliability_bins_for,
)
from gpu_forecasters.calibration.v1.evaluator import evaluate_calibration
from gpu_forecasters.calibration.v1.scoring_rules import (
    brier,
    crps,
    crps_calibration_reward,
)


__all__ = [
    "PREDICTED_BINS",
    "CalibrationDatum",
    "CalibrationReport",
    "LikertNumericMapping",
    "ReliabilityBin",
    "bin_distribution",
    "brier",
    "crps",
    "crps_calibration_reward",
    "entropy",
    "evaluate_calibration",
    "expected_calibration_error",
    "reliability_bins_for",
    "uniform_distribution",
]
