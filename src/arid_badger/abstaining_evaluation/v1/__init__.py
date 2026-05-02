"""Public surface for the abstaining-evaluation library."""

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
from arid_badger.abstaining_evaluation.v1.metrics import (
    aurc,
    decision_set_agreement,
    match_coverage_with_threshold,
    risk_coverage_curve,
    selective_at_coverage,
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
]
