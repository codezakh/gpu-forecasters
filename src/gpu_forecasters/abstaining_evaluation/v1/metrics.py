"""Risk-coverage and selective-prediction metrics.

The curve sweep is parameterized by a sequence of estimates (which
may include ``None`` for parse failures), a ``ConfidenceScore``, and
a ``RiskFunction``. We sweep thresholds at each unique score in the
data so every threshold change moves coverage by at least one row.

``selective_at_coverage`` linearly interpolates and ``aurc`` uses the
trapezoidal rule. Both ignore curve points where risk is NaN, which
happens at coverage 0 (the predicted subset is empty for the pointwise
risks).
"""

from __future__ import annotations

import math
from typing import Sequence

from gpu_forecasters.abstaining_evaluation.v1.confidence_scores import MaxProbScore
from gpu_forecasters.abstaining_evaluation.v1.domain import (
    AbstainDecision,
    ConfidenceScore,
    Predict,
    PredictOrAbstain,
    RiskCoveragePoint,
    RiskFunction,
)
from gpu_forecasters.abstaining_evaluation.v1.threshold_abstainer import (
    ThresholdAbstainPolicy,
)
from gpu_forecasters.eval_dataset_builder.v1 import KernelRuntimeComparison
from gpu_forecasters.landscape_map.v2 import KernelRuntimeEstimate


def risk_coverage_curve(
    *,
    estimates: Sequence[KernelRuntimeEstimate | None],
    comparisons: Sequence[KernelRuntimeComparison],
    score: ConfidenceScore,
    risk: RiskFunction,
) -> list[RiskCoveragePoint]:
    """Sweep thresholds and produce one ``RiskCoveragePoint`` per
    unique score value (plus the boundary points coverage=0 and
    coverage=full-parsed).

    The threshold sweep is monotonic in coverage: as the threshold
    rises, more rows fall below it and abstain. Parse-failure rows
    (``estimate is None``) are forced abstentions at every threshold,
    so the maximum coverage reachable is ``n_parsed / n_total``.
    """
    if len(estimates) != len(comparisons):
        raise ValueError(
            f"estimates and comparisons must align "
            f"(got {len(estimates)} vs {len(comparisons)})"
        )

    parsed_scores: list[float] = [
        score(e) for e in estimates if e is not None
    ]
    # Sweep thresholds at the unique scores in the data plus +inf
    # (which forces every parsed row to abstain → coverage=0). We
    # also include -inf (every parsed row predicts → coverage maximal).
    thresholds = sorted({float("-inf"), *parsed_scores, float("inf")})

    points: list[RiskCoveragePoint] = []
    for tau in thresholds:
        policy = ThresholdAbstainPolicy(score=score, threshold=tau)
        decisions: list[PredictOrAbstain] = [policy(e) for e in estimates]
        n_predicted = sum(1 for d in decisions if isinstance(d, Predict))
        n_abstained = len(decisions) - n_predicted
        coverage = n_predicted / len(decisions)
        r = risk(decisions, comparisons)
        points.append(
            RiskCoveragePoint(
                coverage=coverage,
                risk=r,
                n_predicted=n_predicted,
                n_abstained=n_abstained,
                threshold=tau,
            )
        )
    return points


def aurc(curve: Sequence[RiskCoveragePoint]) -> float:
    """Area under the risk-coverage curve via the trapezoidal rule.

    Drops NaN-risk points (typically the coverage=0 boundary for
    pointwise risks). If two adjacent points share a coverage value
    (a rare boundary case), takes the lower-coverage point's risk.
    """
    valid = [
        p
        for p in curve
        if not (isinstance(p.risk, float) and math.isnan(p.risk))
    ]
    if len(valid) < 2:
        return float("nan")
    by_coverage = sorted(valid, key=lambda p: p.coverage)
    area = 0.0
    for a, b in zip(by_coverage[:-1], by_coverage[1:]):
        if b.coverage == a.coverage:
            continue
        area += 0.5 * (a.risk + b.risk) * (b.coverage - a.coverage)
    return area


def selective_at_coverage(
    curve: Sequence[RiskCoveragePoint], target_coverage: float
) -> float:
    """Linearly interpolate the curve to ``target_coverage``.

    NaN-risk points are dropped before interpolation. Returns NaN if
    no curve point at or below ``target_coverage`` is valid.
    """
    if not 0.0 <= target_coverage <= 1.0:
        raise ValueError(f"target_coverage must be in [0,1], got {target_coverage!r}")
    valid = [
        p
        for p in curve
        if not (isinstance(p.risk, float) and math.isnan(p.risk))
    ]
    if not valid:
        return float("nan")
    by_coverage = sorted(valid, key=lambda p: p.coverage)
    # If target is at or beyond a boundary, snap.
    if target_coverage <= by_coverage[0].coverage:
        return by_coverage[0].risk
    if target_coverage >= by_coverage[-1].coverage:
        return by_coverage[-1].risk
    for a, b in zip(by_coverage[:-1], by_coverage[1:]):
        if a.coverage <= target_coverage <= b.coverage:
            if b.coverage == a.coverage:
                return a.risk
            t = (target_coverage - a.coverage) / (b.coverage - a.coverage)
            return a.risk + t * (b.risk - a.risk)
    return float("nan")


def match_coverage_with_threshold(
    *,
    estimates: Sequence[KernelRuntimeEstimate | None],
    target_coverage: float,
    score: ConfidenceScore,
) -> ThresholdAbstainPolicy:
    """Pick a threshold so the resulting policy's coverage matches a target.

    The threshold is the value such that ``coverage = target_coverage``
    when applied to ``estimates`` under ``score``. Parse failures are
    abstained on, so the maximum reachable coverage is bounded by the
    parsed rate; if ``target_coverage`` exceeds it, we return a
    threshold that simply admits every parsed estimate.
    """
    parsed: list[float] = [score(e) for e in estimates if e is not None]
    n_total = len(estimates)
    if n_total == 0:
        return ThresholdAbstainPolicy(score=score, threshold=float("inf"))
    target_predicted = round(target_coverage * n_total)
    if target_predicted <= 0:
        return ThresholdAbstainPolicy(score=score, threshold=float("inf"))
    if target_predicted >= len(parsed):
        return ThresholdAbstainPolicy(score=score, threshold=float("-inf"))
    # Sort parsed scores descending; the threshold sits between the
    # ``target_predicted``-th and ``(target_predicted + 1)``-th scores
    # (the top-K kept by the threshold).
    sorted_desc = sorted(parsed, reverse=True)
    upper = sorted_desc[target_predicted - 1]
    lower = sorted_desc[target_predicted]
    return ThresholdAbstainPolicy(score=score, threshold=0.5 * (upper + lower))


def decision_set_agreement(
    a: Sequence[PredictOrAbstain],
    b: Sequence[PredictOrAbstain],
) -> float:
    """Jaccard agreement on the abstained subset between two policies.

    ``|a_abstain ∩ b_abstain| / |a_abstain ∪ b_abstain|``. Returns NaN
    if neither policy abstained on any row (the union is empty).
    """
    if len(a) != len(b):
        raise ValueError(
            f"decision lists must align (got {len(a)} vs {len(b)})"
        )
    a_idx = {i for i, d in enumerate(a) if isinstance(d, AbstainDecision)}
    b_idx = {i for i, d in enumerate(b) if isinstance(d, AbstainDecision)}
    union = a_idx | b_idx
    if not union:
        return float("nan")
    return len(a_idx & b_idx) / len(union)


__all__ = [
    "MaxProbScore",  # convenience re-export
    "aurc",
    "decision_set_agreement",
    "match_coverage_with_threshold",
    "risk_coverage_curve",
    "selective_at_coverage",
]
