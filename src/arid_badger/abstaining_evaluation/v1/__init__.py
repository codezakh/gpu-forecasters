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

from arid_badger.abstaining_evaluation.v1.confidence_scores import (
    ConfidenceScore,
    MaxProbScore,
    NegEntropyScore,
    Top2MarginScore,
)
from arid_badger.abstaining_evaluation.v1.domain import (
    AbstainDecision,
    AbstainPolicy,
    Predict,
    PredictOrAbstain,
    RiskCoveragePoint,
    RiskFunction,
    SelectiveMetricRow,
)
from arid_badger.abstaining_evaluation.v1.forecast_reward import (
    ExpectedSpeedupReward,
    ForecastRewardPolicy,
)
from arid_badger.abstaining_evaluation.v1.metrics import (
    aurc,
    decision_set_agreement,
    match_coverage_with_threshold,
    risk_coverage_curve,
    selective_at_coverage,
)
from arid_badger.abstaining_evaluation.v1.mutation_provider import (
    CompoundFeedbackMutationProvider,
    CompoundMutationError,
)
from arid_badger.abstaining_evaluation.v1.observation import (
    CompoundObservation,
    ForecastObservation,
    RealObservation,
)
from arid_badger.abstaining_evaluation.v1.provider import (
    CompoundEvaluationProvider,
)
from arid_badger.abstaining_evaluation.v1.risks import (
    BinaryMismatchRisk,
    RegretRisk,
    SpeedupDistanceRisk,
    bin_midpoint,
)
from arid_badger.abstaining_evaluation.v1.threshold_abstainer import (
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
    "CompoundEvaluationProvider",
    "CompoundFeedbackMutationProvider",
    "CompoundMutationError",
    "CompoundObservation",
    "ExpectedSpeedupReward",
    "ForecastObservation",
    "ForecastRewardPolicy",
    "RealObservation",
]
