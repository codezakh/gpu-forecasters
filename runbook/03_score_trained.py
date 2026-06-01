"""Score a trained-checkpoint surrogate against the canonical eval set.

Sibling of ``01_score_baseline.py`` for the surrogate variants you
trained yourself via ``02_train_surrogate.py``. The trained checkpoint
is read from the ``training_artifact.json`` file that ``02`` writes —
Tinker checkpoint URIs are account-private, so the runbook composes
the two scripts by passing ``02``'s artifact into ``03``'s
``--training-artifact`` flag rather than baking a URI into a config.

Usage::

    # Train first:
    python runbook/02_train_surrogate.py \\
        --config runbook/configs/training/correctness.json \\
        --output-dir runbook_output/02_correctness/

    # Then score that checkpoint:
    python runbook/03_score_trained.py \\
        --config runbook/configs/trained_scoring/correctness.json \\
        --training-artifact runbook_output/02_correctness/training_artifact.json \\
        --output-dir runbook_output/03_correctness/

    # Debug smoke run (1 row per pack, 1 repeat):
    python runbook/03_score_trained.py \\
        --config runbook/configs/trained_scoring/correctness.json \\
        --training-artifact runbook_output/02_correctness__debug/training_artifact.json \\
        --output-dir runbook_output/03_correctness__debug/ \\
        --debug
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from loguru import logger

from gpu_forecasters.cache import FileCache
from gpu_forecasters.experiment_utils import install_loguru_intercept
from gpu_forecasters.runbook import TrainedScoringConfig, apply_debug_overrides, load_canonical_eval_set
from gpu_forecasters.runbook.debug import DEBUG_MAX_ROWS_TRAINED
from gpu_forecasters.runbook.estimators import build_trained_estimator
from gpu_forecasters.runbook.scoring import (
    ScoredComparison,
    ascore_pack_repeat,
    summarize_repeat,
    write_index,
    write_jsonl,
)


def _read_checkpoint_uri(artifact_path: Path) -> str:
    payload = json.loads(artifact_path.read_text())
    uri = payload.get("checkpoint_uri")
    if not isinstance(uri, str) or not uri.startswith("tinker://"):
        raise ValueError(
            f"{artifact_path} does not contain a Tinker checkpoint URI "
            f"(got checkpoint_uri={uri!r}). Run 02_train_surrogate.py "
            f"first to produce a training artifact."
        )
    return uri


async def _arun(
    config: TrainedScoringConfig,
    checkpoint_uri: str,
    output_dir: Path,
    *,
    debug: bool,
) -> None:
    cache: FileCache[ScoredComparison] = FileCache(
        root=output_dir / "cache", value_type=ScoredComparison
    )
    estimator = build_trained_estimator(
        base_model=config.base_model,
        checkpoint_uri=checkpoint_uri,
        renderer_name=config.renderer_name,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
    )
    semaphore = asyncio.Semaphore(config.max_concurrency)

    grouped = load_canonical_eval_set(packs=config.packs)
    logger.info(
        "loaded {} packs from HuggingFace: {}",
        len(grouped),
        sorted(grouped),
    )

    summaries: list[dict[str, object]] = []
    for pack_name in sorted(grouped):
        comparisons = [c for (_row, c) in grouped[pack_name]]
        if debug:
            comparisons = comparisons[:DEBUG_MAX_ROWS_TRAINED]
        logger.info(
            "pack {}: {} rows x {} repeats with checkpoint {}",
            pack_name,
            len(comparisons),
            config.n_repeats,
            checkpoint_uri,
        )

        for repeat in range(config.n_repeats):
            repeat_dir = output_dir / f"repeat_{repeat}" / pack_name
            repeat_dir.mkdir(parents=True, exist_ok=True)

            scored = await ascore_pack_repeat(
                pack_name=pack_name,
                comparisons=comparisons,
                repeat=repeat,
                estimator=estimator,
                cache=cache,
                semaphore=semaphore,
            )
            write_jsonl(repeat_dir / "scored_eval.jsonl", scored)

            summary = summarize_repeat(
                pack_name=pack_name, repeat=repeat, scored=scored
            )
            (repeat_dir / "scored_eval_summary.json").write_text(
                json.dumps(summary, indent=2)
            )
            summaries.append(summary)
            logger.info(
                "done pack={} repeat={}: parsed {}/{}, exact {}/{}",
                pack_name,
                repeat,
                summary["n_parsed"],
                summary["n_total"],
                summary["exact_match"],
                summary["n_total"],
            )

    write_index(
        output_dir=output_dir,
        surrogate_label=config.surrogate_label,
        config_snapshot=config,
        summaries=summaries,
    )
    logger.info("wrote {}", output_dir / "index.json")


def main() -> None:
    install_loguru_intercept()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a JSON config.")
    parser.add_argument(
        "--training-artifact",
        required=True,
        help="Path to training_artifact.json from a 02_train_surrogate.py run.",
    )
    parser.add_argument("--output-dir", required=True, help="Where to write outputs.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Tiny-subset smoke run: 1 row per pack, 1 repeat.",
    )
    args = parser.parse_args()

    config = TrainedScoringConfig.model_validate_json(
        Path(args.config).read_text()
    )
    if args.debug:
        config = apply_debug_overrides(config)

    checkpoint_uri = _read_checkpoint_uri(Path(args.training_artifact))
    logger.info("using trained checkpoint: {}", checkpoint_uri)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(config.model_dump_json(indent=2))
    asyncio.run(_arun(config, checkpoint_uri, output_dir, debug=args.debug))


if __name__ == "__main__":
    main()
