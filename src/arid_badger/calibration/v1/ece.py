"""Reliability-diagram and ECE computation.

For each parsed row we look at the verbalized confidence in the
predicted bin (``bin_confidences[predicted_bin]``), map it through
``LikertNumericMapping`` to a numeric value in ``[0, 1]``, and bucket
it into a fixed grid of confidence bins. Each bucket reports the
average confidence of its members and the empirical accuracy
(``predicted_bin == true_bin``). ECE is the weighted average gap
between mean confidence and accuracy across populated buckets.

Note: this is the "winning-class" calibration view, not full-
distribution calibration. CRPS / Brier capture the latter; ECE is
the standard "if it says high, is it right ~60% of the time?"
sanity check.
"""

from __future__ import annotations

from .domain import (
    CalibrationDatum,
    LikertNumericMapping,
    ReliabilityBin,
)


def _bucket_index(confidence: float, n_buckets: int) -> int:
    """Map a confidence in ``[0, 1]`` to a bucket index in ``[0, n-1]``.

    The right edge ``1.0`` is folded into the last bucket so a
    perfectly confident prediction does not get its own degenerate
    bucket.
    """
    raw = int(confidence * n_buckets)
    return min(raw, n_buckets - 1)


def reliability_bins_for(
    data: list[CalibrationDatum],
    mapping: LikertNumericMapping,
    n_buckets: int = 10,
) -> list[ReliabilityBin]:
    """Bucket parsed rows by the predicted bin's verbalized confidence.

    Unparsed rows (``estimate is None``) are skipped — their
    "predicted bin" is undefined, so they have no reliability-diagram
    contribution. Empty buckets are still emitted so the returned
    list always has length ``n_buckets`` and downstream plotting can
    align on bucket index.
    """
    # (sum_confidence, num_correct, count) per bucket. Stored as a
    # list of mutable lists because tuples can't be updated in place.
    accumulators: list[list[float | int]] = [[0.0, 0, 0] for _ in range(n_buckets)]

    for datum in data:
        if datum.estimate is None:
            continue
        predicted = datum.estimate.predicted_bin
        # Predicted FAILURE on a non-failure-trained surrogate is
        # surprising but possible; skip rather than crash since there
        # is no Likert entry for the failure bin.
        if predicted not in datum.estimate.bin_confidences:
            continue
        confidence = mapping.numeric_for(datum.estimate.bin_confidences[predicted])
        idx = _bucket_index(confidence, n_buckets)
        accumulators[idx][0] = float(accumulators[idx][0]) + confidence
        accumulators[idx][1] = int(accumulators[idx][1]) + (
            1 if predicted == datum.true_bin else 0
        )
        accumulators[idx][2] = int(accumulators[idx][2]) + 1

    out: list[ReliabilityBin] = []
    for i in range(n_buckets):
        sum_conf, num_correct, count = accumulators[i]
        sum_conf_f = float(sum_conf)
        num_correct_i = int(num_correct)
        count_i = int(count)
        lo = i / n_buckets
        hi = (i + 1) / n_buckets
        if count_i > 0:
            out.append(
                ReliabilityBin(
                    confidence_low=lo,
                    confidence_high=hi,
                    mean_confidence=sum_conf_f / count_i,
                    accuracy=num_correct_i / count_i,
                    count=count_i,
                )
            )
        else:
            # Empty bucket: keep the slot so the diagram aligns on bucket index,
            # but report (0, 0) so it plots as a no-op.
            out.append(
                ReliabilityBin(
                    confidence_low=lo,
                    confidence_high=hi,
                    mean_confidence=0.0,
                    accuracy=0.0,
                    count=0,
                )
            )
    return out


def expected_calibration_error(reliability_bins: list[ReliabilityBin]) -> float:
    """Compute ECE from already-bucketed reliability bins.

    ``ECE = Σ_i (n_i / N) * |conf_i - acc_i|``, where the sum is over
    populated buckets. Returns ``0.0`` if there are no parsed rows
    (``N == 0``); the report's ``parsed_rate`` carries the "no signal"
    information in that degenerate case.
    """
    total = sum(b.count for b in reliability_bins)
    if total == 0:
        return 0.0
    return sum(
        (b.count / total) * abs(b.mean_confidence - b.accuracy)
        for b in reliability_bins
        if b.count > 0
    )
