"""End-to-end v2 calibration evaluation."""

from __future__ import annotations

import statistics
from collections.abc import Sequence

from .distribution import entropy, uniform_distribution
from .domain import CalibrationDatum, CalibrationReport
from .ece import expected_calibration_error, reliability_bins_for
from .scoring_rules import brier, crps, nll


def evaluate_calibration(
    data: Sequence[CalibrationDatum],
    n_reliability_buckets: int = 10,
) -> CalibrationReport:
    """Compute aggregate v2 calibration metrics for one held-out set.

    Definitions:

    * ``parsed_rate`` — fraction of rows where the surrogate produced
      a parseable estimate.
    * ``accuracy`` — fraction of *parsed* rows where ``predicted_bin
      == true_bin``.
    * ``mean_entropy`` — mean Shannon entropy (nats) of the
      simplex over parsed rows.
    * ``ece`` — winning-class ECE over parsed rows, bucketed by
      ``bin_probabilities[predicted_bin]``.
    * ``mean_brier_parsed`` / ``mean_crps_parsed`` /
      ``mean_nll_parsed`` — proper scoring rules averaged over parsed
      rows only.
    * ``mean_brier`` / ``mean_crps`` / ``mean_nll`` — full-set means
      with parse failures penalized by a uniform fallback (so a model
      can't dodge the score by failing to answer).
    * ``mean_raw_probability_sum`` — mean of the model's pre-
      renormalization probability sum across parsed rows. A v2-
      specific calibration-health signal: values far from 1 indicate
      the model is bad at producing simplexes even when the
      renormalized distribution is well-formed.
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
            mean_nll=0.0,
            mean_nll_parsed=0.0,
            mean_raw_probability_sum=None,
            reliability_bins=[],
        )

    parsed = [d for d in data if d.estimate is not None]
    n_parsed = len(parsed)

    n_correct = sum(
        1
        for d in parsed
        if d.estimate is not None and d.estimate.predicted_bin == d.true_bin
    )
    accuracy = n_correct / n_parsed if n_parsed > 0 else 0.0

    if n_parsed > 0:
        total_entropy = 0.0
        for d in parsed:
            assert d.estimate is not None
            total_entropy += entropy(d.estimate.bin_probabilities)
        mean_entropy = total_entropy / n_parsed
    else:
        mean_entropy = 0.0

    parsed_brier_total = 0.0
    parsed_crps_total = 0.0
    parsed_nll_total = 0.0
    full_brier_total = 0.0
    full_crps_total = 0.0
    full_nll_total = 0.0
    fallback = uniform_distribution()
    for d in data:
        if d.estimate is None:
            full_brier_total += brier(fallback, d.true_bin)
            full_crps_total += crps(fallback, d.true_bin)
            full_nll_total += nll(fallback, d.true_bin)
            continue
        b = brier(d.estimate.bin_probabilities, d.true_bin)
        c = crps(d.estimate.bin_probabilities, d.true_bin)
        n = nll(d.estimate.bin_probabilities, d.true_bin)
        parsed_brier_total += b
        parsed_crps_total += c
        parsed_nll_total += n
        full_brier_total += b
        full_crps_total += c
        full_nll_total += n

    mean_brier_parsed = parsed_brier_total / n_parsed if n_parsed > 0 else 0.0
    mean_crps_parsed = parsed_crps_total / n_parsed if n_parsed > 0 else 0.0
    mean_nll_parsed = parsed_nll_total / n_parsed if n_parsed > 0 else 0.0
    mean_brier = full_brier_total / n_total
    mean_crps = full_crps_total / n_total
    mean_nll = full_nll_total / n_total

    raw_sums = [
        d.estimate.raw_probability_sum
        for d in parsed
        if d.estimate is not None
    ]
    mean_raw_sum = float(statistics.mean(raw_sums)) if raw_sums else None

    reliability = reliability_bins_for(data, n_reliability_buckets)
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
        mean_nll=mean_nll,
        mean_nll_parsed=mean_nll_parsed,
        mean_raw_probability_sum=mean_raw_sum,
        reliability_bins=reliability,
    )
