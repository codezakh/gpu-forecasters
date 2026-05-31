"""Public surface for the abstaining-evaluation library.

Two surfaces share this module:

* The flat-eval thrust (ZAI-71) — measure a surrogate as an
  abstaining classifier on a held-out set with no search. Exports
  ``Predict`` / ``AbstainDecision`` (policy decisions),
  ``ConfidenceScore`` / ``RiskFunction`` and their concretes,
  threshold abstainer, and risk-coverage / AURC metrics.

* The search-embedded thrust (ZAI-72) — drop an abstaining surrogate
  behind the v2 search's evaluation provider seam. Exports
  ``CompoundObservation`` (observation discriminated union),
  ``ForecastRewardPolicy`` / ``ExpectedSpeedupReward``,
  ``CompoundEvaluationProvider``, and
  ``CompoundFeedbackMutationProvider``.
"""

from gpu_forecasters.abstaining_evaluation.v1.confidence_scores import (
    ConfidenceScore,
    MaxProbScore,
    NegEntropyScore,
    Top2MarginScore,
)
from gpu_forecasters.abstaining_evaluation.v1.domain import (
    AbstainDecision,
    AbstainPolicy,
    Predict,
    PredictOrAbstain,
    RiskCoveragePoint,
    RiskFunction,
    SelectiveMetricRow,
)
from gpu_forecasters.abstaining_evaluation.v1.forecast_checking import (
    CheckedForecast,
    ForecastChecker,
    forecasts_to_check,
    load_checked_forecasts,
)
from gpu_forecasters.abstaining_evaluation.v1.forecast_reward import (
    ExpectedSpeedupReward,
    ForecastRewardPolicy,
)
from gpu_forecasters.abstaining_evaluation.v1.metrics import (
    aurc,
    decision_set_agreement,
    match_coverage_with_threshold,
    risk_coverage_curve,
    selective_at_coverage,
)
from gpu_forecasters.abstaining_evaluation.v1.mutation_provider import (
    CompoundFeedbackMutationProvider,
    CompoundMutationError,
)
from gpu_forecasters.abstaining_evaluation.v1.observation import (
    CompoundObservation,
    ForecastObservation,
    RealObservation,
)
from gpu_forecasters.abstaining_evaluation.v1.provider import (
    CompoundEvaluationProvider,
)
from gpu_forecasters.abstaining_evaluation.v1.risks import (
    BinaryMismatchRisk,
    RegretRisk,
    SpeedupDistanceRisk,
    bin_midpoint,
)
from gpu_forecasters.abstaining_evaluation.v1.threshold_abstainer import (
    ThresholdAbstainPolicy,
)


__all__ = [
    # Flat-eval surface (ZAI-71)
    "AbstainDecision",
    "AbstainPolicy",
    "BinaryMismatchRisk",
    "ConfidenceScore",
    "MaxProbScore",
    "NegEntropyScore",
    "Predict",
    "PredictOrAbstain",
    "RegretRisk",
    "RiskCoveragePoint",
    "RiskFunction",
    "SelectiveMetricRow",
    "SpeedupDistanceRisk",
    "ThresholdAbstainPolicy",
    "Top2MarginScore",
    "aurc",
    "bin_midpoint",
    "decision_set_agreement",
    "match_coverage_with_threshold",
    "risk_coverage_curve",
    "selective_at_coverage",
    # Search-embedded surface (ZAI-72)
    "CheckedForecast",
    "CompoundEvaluationProvider",
    "CompoundFeedbackMutationProvider",
    "CompoundMutationError",
    "CompoundObservation",
    "ExpectedSpeedupReward",
    "ForecastChecker",
    "ForecastObservation",
    "ForecastRewardPolicy",
    "RealObservation",
    "forecasts_to_check",
    "load_checked_forecasts",
]
