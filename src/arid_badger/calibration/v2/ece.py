"""Winning-class Expected Calibration Error for v2 surrogates.

Buckets parsed rows by ``bin_probabilities[predicted_bin]`` (the
model's own claimed confidence in its argmax) and folds into a scalar
``Σ (count_i / total) · |conf_i − acc_i|``.
"""

from __future__ import annotations

from collections.abc import Sequence

from .domain import CalibrationDatum, ReliabilityBin


def _bucket_index(confidence: float, n_buckets: int) -> int:
    raw = int(confidence * n_buckets)
    return min(raw, n_buckets - 1)


def reliability_bins_for(
    data: Sequence[CalibrationDatum], n_buckets: int = 10
) -> list[ReliabilityBin]:
    """Bucket parsed rows by ``bin_probabilities[predicted_bin]``.

    Empty buckets are kept (count=0, mean_confidence=accuracy=0) so
    the diagram aligns on bucket index across surrogates.
    """
    accumulators: list[list[float | int]] = [
        [0.0, 0, 0] for _ in range(n_buckets)
    ]
    for d in data:
        if d.estimate is None:
            continue
        confidence = d.estimate.bin_probabilities[d.estimate.predicted_bin]
        idx = _bucket_index(confidence, n_buckets)
        accumulators[idx][0] = float(accumulators[idx][0]) + confidence
        accumulators[idx][1] = int(accumulators[idx][1]) + (
            1 if d.estimate.predicted_bin == d.true_bin else 0
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
    total = sum(b.count for b in reliability_bins)
    if total == 0:
        return 0.0
    return sum(
        (b.count / total) * abs(b.mean_confidence - b.accuracy)
        for b in reliability_bins
        if b.count > 0
    )
