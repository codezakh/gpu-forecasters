"""Run one upstream PUCT search end-to-end.

Reproduces one of the source PUCT searches that produced the raw data
archive used downstream — eval-set rows, RL training pool, discovery
pairs. Costs real GPU time on Modal at the configured shape; the
``--debug`` flag clamps the search to one step / one mutation so the
plumbing can be exercised without paying for a full search.

The mutator is the LLM that proposes kernel edits (Gemini-3 Flash by
default). The evaluator runs the proposed kernels on Modal against
the pack's benchmark cases. The PUCT driver keeps an event-sourced
log under ``<output_dir>/run_00/events.jsonl`` for crash-safe resume.

Usage::

    python runbook/00_upstream_puct.py \\
        --config runbook/configs/upstream_puct/gpu_mode/trimul__e0033__gemini3_flash.json \\
        --output-dir runbook_output/00_trimul_search/

    # Plumbing-only smoke run (1 step, 1 parent, 1 mutation):
    python runbook/00_upstream_puct.py \\
        --config runbook/configs/upstream_puct/gpu_mode/trimul__e0033__gemini3_flash.json \\
        --output-dir runbook_output/00_trimul_search__debug/ \\
        --debug
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from gpu_forecasters.experiment_utils import install_loguru_intercept
from gpu_forecasters.gpu_mode_kernel.experiment_helper import (
    ExperimentConfig,
    ProviderConfig,
    RunConfig,
    run_pack_experiment,
)
from gpu_forecasters.max_reward_puct.v2.config import SearchConfig as V2SearchConfig
from gpu_forecasters.runbook import UpstreamPuctConfig, apply_debug_overrides
from gpu_forecasters.runbook.packs import get_pack_runtime_and_case_type


def _to_library_config(config: UpstreamPuctConfig) -> ExperimentConfig:
    """Project the runbook config into the library's ``ExperimentConfig``."""
    return ExperimentConfig(
        num_runs=1,
        run=RunConfig(
            search=V2SearchConfig(
                total_budget_steps=config.total_budget_steps,
                batch_size=config.batch_size,
                samples_per_parent=config.samples_per_parent,
                k_per_parent=config.k_per_parent,
                archive_capacity=config.archive_capacity,
                c_puct=config.c_puct,
                per_request_timeout_s=config.request_timeout_s,
            ),
            provider=ProviderConfig(
                model_slug=config.mutator_model_slug,
                gpu=config.hardware.gpu,
                aggregator=config.aggregator,
                max_llm_concurrency=config.max_llm_concurrency,
                max_tokens=None,
                request_timeout_s=config.request_timeout_s,
            ),
        ),
    )


def main() -> None:
    install_loguru_intercept()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a JSON config.")
    parser.add_argument("--output-dir", required=True, help="Where to write outputs.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Plumbing-only smoke run: 1 step, 1 parent, 1 mutation.",
    )
    args = parser.parse_args()

    config = UpstreamPuctConfig.model_validate_json(Path(args.config).read_text())
    if args.debug:
        config = apply_debug_overrides(config)
        logger.info(
            "debug mode: steps={}, batch={}, spp={}, k={}",
            config.total_budget_steps,
            config.batch_size,
            config.samples_per_parent,
            config.k_per_parent,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(config.model_dump_json(indent=2))

    pack_runtime, case_speedup_type = get_pack_runtime_and_case_type(config.pack)
    lib_config = _to_library_config(config)

    run_pack_experiment(
        pack_runtime=pack_runtime,
        case_speedup_type=case_speedup_type,
        output_dir=output_dir,
        config=lib_config,
    )


if __name__ == "__main__":
    main()
