"""Score one surrogate against the discovery-pair eval set.

Reproduces §4.5 of the paper for one surrogate. Each row in
``codezakh/gpu-forecasters-discovery-pairs`` is a parent→child kernel
pair: the surrogate is asked for the child's speedup *relative to the
parent*, not relative to the original reference. Outputs use the same
per-(repeat, problem) layout as ``01_score_baseline.py`` so the §4.5
collation reads them unchanged.

Usage::

    python runbook/05_score_discovery.py \\
        --config runbook/configs/discovery_scoring/gemini3_flash.json \\
        --output-dir runbook_output/05_discovery_gemini3_flash/

    # Debug-only smoke run (5 pairs, 1 repeat, 1 concurrent call):
    python runbook/05_score_discovery.py \\
        --config runbook/configs/discovery_scoring/gemini3_flash.json \\
        --output-dir runbook_output/05_discovery_gemini3_flash__debug/ \\
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
from gpu_forecasters.runbook import (
    DiscoveryScoringConfig,
    apply_debug_overrides,
    build_estimator_from_backend,
    load_discovery_pairs,
)
from gpu_forecasters.runbook.debug import DEBUG_MAX_ROWS_DISCOVERY
from gpu_forecasters.runbook.scoring import (
    ScoredComparison,
    ascore_pack_repeat,
    summarize_repeat,
    write_index,
    write_jsonl,
)


async def _arun(config: DiscoveryScoringConfig, output_dir: Path, *, debug: bool) -> None:
    cache: FileCache[ScoredComparison] = FileCache(
        root=output_dir / "cache", value_type=ScoredComparison
    )
    estimator = build_estimator_from_backend(config.backend)
    semaphore = asyncio.Semaphore(config.max_concurrency)

    grouped = load_discovery_pairs(families=config.benchmark_families)
    total = sum(len(v) for v in grouped.values())
    logger.info(
        "loaded {} discovery pairs across {} problems: families={}",
        total,
        len(grouped),
        config.benchmark_families,
    )

    if debug:
        # Take a single problem id and cap to N rows.
        first_problem = sorted(grouped)[0]
        grouped = {
            first_problem: grouped[first_problem][:DEBUG_MAX_ROWS_DISCOVERY]
        }
        logger.info(
            "debug mode: scoring problem={} with {} rows",
            first_problem,
            len(grouped[first_problem]),
        )

    summaries: list[dict[str, object]] = []
    for problem_id in sorted(grouped):
        comparisons = [c for (_row, c) in grouped[problem_id]]
        logger.info(
            "problem {}: {} rows x {} repeats with surrogate {!r}",
            problem_id,
            len(comparisons),
            config.n_repeats,
            config.surrogate_label,
        )

        for repeat in range(config.n_repeats):
            repeat_dir = output_dir / f"repeat_{repeat}" / problem_id
            repeat_dir.mkdir(parents=True, exist_ok=True)

            scored = await ascore_pack_repeat(
                pack_name=problem_id,
                comparisons=comparisons,
                repeat=repeat,
                estimator=estimator,
                cache=cache,
                semaphore=semaphore,
            )
            write_jsonl(repeat_dir / "scored_eval.jsonl", scored)

            summary = summarize_repeat(
                pack_name=problem_id, repeat=repeat, scored=scored
            )
            (repeat_dir / "scored_eval_summary.json").write_text(
                json.dumps(summary, indent=2)
            )
            summaries.append(summary)
            logger.info(
                "done problem={} repeat={}: parsed {}/{}, exact {}/{}",
                problem_id,
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
    parser.add_argument("--output-dir", required=True, help="Where to write outputs.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Tiny-subset smoke run: 5 pairs, 1 repeat, 1 concurrent call.",
    )
    args = parser.parse_args()

    config = DiscoveryScoringConfig.model_validate_json(
        Path(args.config).read_text()
    )
    if args.debug:
        config = apply_debug_overrides(config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(config.model_dump_json(indent=2))
    asyncio.run(_arun(config, output_dir, debug=args.debug))


if __name__ == "__main__":
    main()
