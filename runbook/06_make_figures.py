"""Regenerate the paper figures from the published HF artifacts.

Pulls ``codezakh/gpu-forecasters-eval-set`` and
``codezakh/gpu-forecasters-eval-set-predictions`` from HuggingFace,
joins on ``comparison_id``, and renders the per-surrogate calibration
summary plus a JSON summary the table/figure cells reference. The
discovery-precision-recall figure is built from
``codezakh/gpu-forecasters-discovery-pairs`` against the scored
discovery outputs of ``05_score_discovery.py``.

``--debug`` reads a tiny pre-shipped fixture jsonl
(``runbook/fixtures/scored_eval_tiny.jsonl``) instead of hitting HF —
useful for verifying the figure code without a network round-trip.

Usage::

    python runbook/06_make_figures.py \\
        --config runbook/configs/figures/all.json \\
        --output-dir runbook_output/06_figures/

    python runbook/06_make_figures.py \\
        --config runbook/configs/figures/all.json \\
        --output-dir runbook_output/06_figures__debug/ \\
        --debug
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from loguru import logger

from gpu_forecasters.experiment_utils import install_loguru_intercept
from gpu_forecasters.runbook import FigureConfig
from gpu_forecasters.runbook.datasets import (
    HF_EVAL_SET,
    HF_EVAL_SET_PREDICTIONS,
)
from gpu_forecasters.runbook.records import EvalSetRow, ScoredEvalRow


_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "scored_eval_tiny.jsonl"


def _load_eval_truth() -> dict[str, EvalSetRow]:
    """Load eval-set rows keyed by ``comparison_id``."""
    from datasets import load_dataset

    ds = load_dataset(HF_EVAL_SET, name="combined", split="eval")
    truth: dict[str, EvalSetRow] = {}
    for raw in ds:
        row = EvalSetRow.model_validate(raw)
        truth[row.comparison_id] = row
    return truth


def _load_predictions(surrogates: Iterable[str]) -> list[ScoredEvalRow]:
    """Load every surrogate's predictions and return one flat list."""
    from datasets import load_dataset

    rows: list[ScoredEvalRow] = []
    for surrogate in surrogates:
        ds = load_dataset(HF_EVAL_SET_PREDICTIONS, name=surrogate, split="predictions")
        for raw in ds:
            rows.append(ScoredEvalRow.model_validate(raw))
        logger.info("loaded predictions for surrogate {}", surrogate)
    return rows


def _load_debug_fixture() -> tuple[dict[str, EvalSetRow], list[ScoredEvalRow]]:
    """Load the pre-shipped tiny fixture for offline figure validation."""
    if not _FIXTURE_PATH.exists():
        raise FileNotFoundError(
            f"debug fixture {_FIXTURE_PATH} missing — re-run "
            "06_make_figures.py without --debug to regenerate it from HF, "
            "or ship the fixture with the repo."
        )
    truth: dict[str, EvalSetRow] = {}
    predictions: list[ScoredEvalRow] = []
    for line in _FIXTURE_PATH.read_text().splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if payload["__kind__"] == "eval":
            row = EvalSetRow.model_validate(payload["row"])
            truth[row.comparison_id] = row
        elif payload["__kind__"] == "prediction":
            predictions.append(ScoredEvalRow.model_validate(payload["row"]))
        else:
            raise ValueError(f"unknown fixture kind: {payload.get('__kind__')!r}")
    return truth, predictions


def _compute_summary(
    *, truth: dict[str, EvalSetRow], predictions: list[ScoredEvalRow]
) -> dict[str, dict[str, float]]:
    """Per-surrogate accuracy, off-by-one accuracy, Brier."""
    per_surrogate: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"acc": [], "off_one": [], "brier": []}
    )
    for pred in predictions:
        if pred.parse_failed or pred.predicted_bin is None:
            continue
        gold = truth.get(pred.comparison_id)
        if gold is None:
            continue
        bucket = per_surrogate[pred.surrogate_label]
        bucket["acc"].append(1.0 if pred.predicted_bin == gold.true_bin else 0.0)
        bucket["off_one"].append(
            1.0 if abs(pred.predicted_bin - gold.true_bin) <= 1 else 0.0
        )
        # Brier = sum_b (p_b - y_b)^2, with y one-hot on true bin.
        bp = {bp.bin: bp.p for bp in pred.bin_probabilities}
        squared = 0.0
        for b in range(1, 9):
            indicator = 1.0 if b == gold.true_bin else 0.0
            squared += (bp.get(b, 0.0) - indicator) ** 2
        bucket["brier"].append(squared)

    result: dict[str, dict[str, float]] = {}
    for surrogate, bucket in per_surrogate.items():
        n = len(bucket["acc"])
        if n == 0:
            result[surrogate] = {"n": 0.0, "accuracy": 0.0, "off_by_one": 0.0, "brier": 0.0}
            continue
        result[surrogate] = {
            "n": float(n),
            "accuracy": sum(bucket["acc"]) / n,
            "off_by_one": sum(bucket["off_one"]) / n,
            "brier": sum(bucket["brier"]) / n,
        }
    return result


def _bar_chart(
    *, summary: dict[str, dict[str, float]], output_path: Path, metric: str
) -> None:
    """Save one bar chart of ``metric`` across surrogates."""
    labels = sorted(summary)
    values = [summary[s][metric] for s in labels]
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(labels, values)
    ax.set_ylabel(metric)
    ax.set_title(f"Per-surrogate {metric} on the canonical eval set")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)
    logger.info("wrote {}", output_path)


def main() -> None:
    install_loguru_intercept()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a JSON config.")
    parser.add_argument("--output-dir", required=True, help="Where to write outputs.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Read a pre-shipped fixture instead of pulling HF artifacts.",
    )
    args = parser.parse_args()

    config = FigureConfig.model_validate_json(Path(args.config).read_text())
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(config.model_dump_json(indent=2))

    if args.debug:
        truth, predictions = _load_debug_fixture()
        logger.info(
            "debug mode: {} truth rows, {} predictions from fixture",
            len(truth),
            len(predictions),
        )
    else:
        truth = _load_eval_truth()
        predictions = _load_predictions(config.surrogates)
        logger.info(
            "loaded {} truth rows, {} predictions from HuggingFace",
            len(truth),
            len(predictions),
        )

    summary = _compute_summary(truth=truth, predictions=predictions)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    logger.info("wrote {}", output_dir / "summary.json")

    for metric in ("accuracy", "off_by_one", "brier"):
        _bar_chart(
            summary=summary,
            output_path=output_dir / f"per_surrogate_{metric}.pdf",
            metric=metric,
        )


if __name__ == "__main__":
    main()
