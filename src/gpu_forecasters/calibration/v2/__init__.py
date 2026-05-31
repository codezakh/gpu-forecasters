"""Calibration metrics for v2 (numerical-simplex) surrogates.

See ``README.md`` for the design overview. Public surface re-exported
here so callers can write ``from gpu_forecasters.calibration.v2 import
evaluate_calibration``.
"""

from gpu_forecasters.calibration.v2.distribution import (
    entropy,
    uniform_distribution,
)
from gpu_forecasters.calibration.v2.domain import (
    CalibrationDatum,
    CalibrationReport,
    ReliabilityBin,
)
from gpu_forecasters.calibration.v2.ece import (
    expected_calibration_error,
    reliability_bins_for,
)
from gpu_forecasters.calibration.v2.evaluator import evaluate_calibration
from gpu_forecasters.calibration.v2.scoring_rules import (
    brier,
    crps,
    nll,
)


__all__ = [
    "CalibrationDatum",
    "CalibrationReport",
    "ReliabilityBin",
    "brier",
    "crps",
    "entropy",
    "evaluate_calibration",
    "expected_calibration_error",
    "nll",
    "reliability_bins_for",
    "uniform_distribution",
]
