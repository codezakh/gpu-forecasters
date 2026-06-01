"""Score one off-the-shelf surrogate against the canonical eval set.

Reproduces §4.3 of the paper for one off-the-shelf model (Gemini-3
Flash, GPT-OSS-120B on Together, GPT-OSS-20B on Tinker, DeepSeek-V4).
Picks the surrogate via the JSON config under
``configs/baseline_scoring/``.

Usage::

    python runbook/01_score_baseline.py \\
        --config runbook/configs/baseline_scoring/gemini3_flash.json \\
        --output-dir runbook_output/01_gemini3_flash/

    # Debug-only smoke run (1 row per pack, 1 repeat, 1 concurrent call):
    python runbook/01_score_baseline.py \\
        --config runbook/configs/baseline_scoring/gemini3_flash.json \\
        --output-dir runbook_output/01_gemini3_flash__debug/ \\
        --debug

Output layout::

    <output_dir>/
        config.json
        cache/                       # crash-safe cache keyed by (repeat, pack, row)
        repeat_0/
            <pack>/scored_eval.jsonl
            <pack>/scored_eval_summary.json
        repeat_1/
            ...
        index.json
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

from loguru import logger

from gpu_forecasters.cache import FileCache
from gpu_forecasters.experiment_utils import install_loguru_intercept
from gpu_forecasters.runbook import (
    BaselineScoringConfig,
    apply_debug_overrides,
    build_estimator_from_backend,
    load_canonical_eval_set,
)
from gpu_forecasters.runbook.debug import DEBUG_MAX_ROWS_PER_PACK_BASELINE
from gpu_forecasters.runbook.scoring import (
    ScoredComparison,
    ascore_pack_repeat,
    summarize_repeat,
    write_index,
    write_jsonl,
)


async def _arun(config: BaselineScoringConfig, output_dir: Path, *, debug: bool) -> None:
    cache: FileCache[ScoredComparison] = FileCache(
        root=output_dir / "cache", value_type=ScoredComparison
    )
    estimator = build_estimator_from_backend(config.backend)
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
            comparisons = comparisons[:DEBUG_MAX_ROWS_PER_PACK_BASELINE]
        logger.info(
            "pack {}: {} rows x {} repeats with surrogate {!r}",
            pack_name,
            len(comparisons),
            config.n_repeats,
            config.surrogate_label,
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
                "done pack={} repeat={}: parsed {}/{}, exact {}/{}, mean_elapsed={:.1f}s",
                pack_name,
                repeat,
                summary["n_parsed"],
                summary["n_total"],
                summary["exact_match"],
                summary["n_total"],
                summary["mean_elapsed_s"],
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
    parser.add_argument("--output-dir", required=True, help="Where to write outputs.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Tiny-subset smoke run: 1 row per pack, 1 repeat, 1 concurrent call.",
    )
    args = parser.parse_args()

    config = BaselineScoringConfig.model_validate_json(
        Path(args.config).read_text()
    )
    if args.debug:
        config = apply_debug_overrides(config)
        logger.info("debug mode: overrode n_repeats={}, max_concurrency={}", config.n_repeats, config.max_concurrency)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(config.model_dump_json(indent=2))
    asyncio.run(_arun(config, output_dir, debug=args.debug))


if __name__ == "__main__":
    main()
