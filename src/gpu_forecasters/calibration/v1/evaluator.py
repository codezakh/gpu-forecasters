"""Top-level entry point: a list of held-out triples → ``CalibrationReport``.

The evaluator is a pure function: it does not load files, does not
talk to a surrogate, and does not write artifacts. Callers are
responsible for assembling the ``CalibrationDatum`` list (typically
by loading scored eval rows from a previous experiment's workspace).
"""

from __future__ import annotations

from .distribution import bin_distribution, entropy, uniform_distribution
from .domain import (
    CalibrationDatum,
    CalibrationReport,
    LikertNumericMapping,
)
from .ece import expected_calibration_error, reliability_bins_for
from .scoring_rules import brier, crps


def evaluate_calibration(
    data: list[CalibrationDatum],
    mapping: LikertNumericMapping = LikertNumericMapping(),
    n_reliability_buckets: int = 10,
) -> CalibrationReport:
    """Aggregate per-row calibration metrics over a held-out set.

    Definitions (also documented on ``CalibrationReport``):

    * ``parsed_rate`` — fraction of rows where the surrogate produced
      a parseable estimate.
    * ``accuracy`` — fraction of *parsed* rows where ``predicted_bin
      == true_bin``.
    * ``mean_entropy`` — average Shannon entropy (nats) of the
      verbalized 8-bin distribution over parsed rows.
    * ``ece`` — Expected Calibration Error over parsed rows, bucketed
      by the predicted bin's verbalized confidence.
    * ``mean_brier_parsed`` / ``mean_crps_parsed`` — proper scoring
      rules averaged over parsed rows only.
    * ``mean_brier`` / ``mean_crps`` — proper scoring rules over the
      full set, with parse failures penalized by a uniform fallback
      distribution. Reported alongside the parsed-only versions so
      a model that fails-to-answer cannot dodge its scoring-rule
      penalty.
    """
    n_total = len(data)
    if n_total == 0:
        return CalibrationReport(
            n_total=0,
            n_parsed=0,
            parsed_rate=0.0,
            accuracy=0.0,
            mean_entropy=0.0,
            ece=0.0,
            mean_brier=0.0,
            mean_brier_parsed=0.0,
            mean_crps=0.0,
            mean_crps_parsed=0.0,
            reliability_bins=[],
            likert_mapping=mapping,
        )

    parsed_rows = [d for d in data if d.estimate is not None]
    n_parsed = len(parsed_rows)

    # Argmax-bin accuracy over parsed rows.
    n_correct = sum(
        1
        for d in parsed_rows
        if d.estimate is not None and d.estimate.predicted_bin == d.true_bin
    )
    accuracy = n_correct / n_parsed if n_parsed > 0 else 0.0

    # Sharpness: mean entropy of the predicted distribution over parsed rows.
    if n_parsed > 0:
        total_entropy = 0.0
        for d in parsed_rows:
            assert d.estimate is not None  # narrowed by filter above
            dist = bin_distribution(d.estimate, mapping)
            total_entropy += entropy(dist)
        mean_entropy = total_entropy / n_parsed
    else:
        mean_entropy = 0.0

    # Proper scoring rules. Parse failures are scored against uniform.
    parsed_brier_total = 0.0
    parsed_crps_total = 0.0
    full_brier_total = 0.0
    full_crps_total = 0.0
    fallback = uniform_distribution()
    for d in data:
        if d.estimate is None:
            full_brier_total += brier(fallback, d.true_bin)
            full_crps_total += crps(fallback, d.true_bin)
            continue
        dist = bin_distribution(d.estimate, mapping)
        b = brier(dist, d.true_bin)
        c = crps(dist, d.true_bin)
        parsed_brier_total += b
        parsed_crps_total += c
        full_brier_total += b
        full_crps_total += c

    mean_brier_parsed = parsed_brier_total / n_parsed if n_parsed > 0 else 0.0
    mean_crps_parsed = parsed_crps_total / n_parsed if n_parsed > 0 else 0.0
    mean_brier = full_brier_total / n_total
    mean_crps = full_crps_total / n_total

    reliability = reliability_bins_for(data, mapping, n_buckets=n_reliability_buckets)
    ece = expected_calibration_error(reliability)

    return CalibrationReport(
        n_total=n_total,
        n_parsed=n_parsed,
        parsed_rate=n_parsed / n_total,
        accuracy=accuracy,
        mean_entropy=mean_entropy,
        ece=ece,
        mean_brier=mean_brier,
        mean_brier_parsed=mean_brier_parsed,
        mean_crps=mean_crps,
        mean_crps_parsed=mean_crps_parsed,
        reliability_bins=reliability,
        likert_mapping=mapping,
    )
