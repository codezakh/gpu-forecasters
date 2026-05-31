"""Reliability-diagram rendering.

Pulled out of the evaluator so the pure scoring path has no
matplotlib dependency in tests, and so callers that only need the
scalar metrics never load matplotlib.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

from .domain import CalibrationReport


def render_reliability_diagram(
    report: CalibrationReport, title: str, output_path: Path
) -> None:
    """Save a reliability diagram for one report to ``output_path``.

    Bars: empirical accuracy per confidence bucket. Diagonal line:
    perfect calibration. Bucket counts annotated above each bar so the
    reader can tell apart "well-calibrated bucket" from
    "well-calibrated bucket with three points in it".
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    bins = report.reliability_bins
    centers = [(b.confidence_low + b.confidence_high) / 2 for b in bins]
    accuracies = [b.accuracy for b in bins]
    counts = [b.count for b in bins]
    width = 1.0 / len(bins) if bins else 0.0

    _ = ax.bar(
        centers,
        accuracies,
        width=width * 0.9,
        edgecolor="black",
        color="#4C72B0",
        label="empirical accuracy",
    )
    _ = ax.plot([0, 1], [0, 1], color="gray", linestyle="--", label="perfect")
    for c, a, n in zip(centers, accuracies, counts):
        if n > 0:
            _ = ax.text(
                c, a + 0.02, str(n), ha="center", va="bottom", fontsize=8
            )
    _ = ax.set_xlim(0, 1)
    _ = ax.set_ylim(0, 1.1)
    _ = ax.set_xlabel("verbalized confidence in predicted bin")
    _ = ax.set_ylabel("empirical accuracy")
    _ = ax.set_title(
        f"{title}\nECE={report.ece:.3f}  "
        f"acc={report.accuracy:.3f}  "
        f"H̄={report.mean_entropy:.2f}  "
        f"CRPS={report.mean_crps:.3f}"
    )
    _ = ax.legend(loc="upper left")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, bbox_inches="tight", dpi=150)
    plt.close(fig)
