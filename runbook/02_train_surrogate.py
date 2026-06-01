"""Train one of the three published surrogate variants via GRPO.

Reproduces e0158 (correctness), e0159 (correctness + Brier), or e0160
(correctness + CRPS) depending on which config you pass. Training
data is read from ``codezakh/gpu-forecasters-rl-training-pool`` on
HuggingFace.

This script kicks off a real Tinker training run by default — the
``--debug`` flag clamps it to one iteration / two samples / 1024
tokens / four training rows so you can verify the plumbing without
burning compute.

Usage::

    python runbook/02_train_surrogate.py \\
        --config runbook/configs/training/correctness.json \\
        --output-dir runbook_output/02_correctness/

    python runbook/02_train_surrogate.py \\
        --config runbook/configs/training/correctness.json \\
        --output-dir runbook_output/02_correctness__debug/ \\
        --debug

Output layout::

    <output_dir>/
        training_config.json
        training/                    # cookbook's training logs (metrics.jsonl etc.)
        training_artifact.json       # checkpoint URI + token usage + config snapshot
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from gpu_forecasters.experiment_utils import install_loguru_intercept
from gpu_forecasters.landscape_map.v1.domain import HardwareContext as HardwareContextV1
from gpu_forecasters.landscape_map.v2 import (
    HardwareContext as HardwareContextV2,
    LabeledKernelItem,
)
from gpu_forecasters.runbook import (
    TrainingRunConfig,
    apply_debug_overrides,
    load_rl_training_pool,
)
from gpu_forecasters.runbook.debug import DEBUG_MAX_TRAINING_ROWS
from gpu_forecasters.runbook.records import RlTrainingRow
from gpu_forecasters.runbook.training import train


_DEFAULT_HARDWARE_V1 = HardwareContextV1(
    device_name="NVIDIA A100-SXM4-80GB",
    compute_capability=(8, 0),
    total_global_memory_gb=80.0,
    multiprocessor_count=108,
    max_threads_per_multiprocessor=2048,
    clock_rate_ghz=1.41,
    memory_clock_rate_ghz=1.512,
    memory_bus_width_bits=5120,
)


def _row_to_item(row: RlTrainingRow) -> LabeledKernelItem:
    """Project an HF training-pool row onto the library-side training item.

    The HF row carries a single ``hardware`` string. Every training row
    in the paper is on the A100-80GB SXM4, so we hand the v2 surrogate
    that full ``HardwareContext`` after a v1→v2 round-trip.
    """
    if row.hardware != _DEFAULT_HARDWARE_V1.device_name:
        raise ValueError(
            f"Unexpected hardware {row.hardware!r}; the runbook only ships "
            f"the A100-SXM4-80GB hardware profile. Add the new profile to "
            f"runbook/02_train_surrogate.py if a new device joins the paper."
        )
    hardware_v2 = HardwareContextV2(**_DEFAULT_HARDWARE_V1.model_dump())
    return LabeledKernelItem(
        pack_name=row.pack,
        anchor_source=row.pair_type,
        anchor_code=row.anchor_code,
        candidate_code=row.candidate_code,
        speedup_geomean=row.aggregated_speedup,
        hardware=hardware_v2,
        source_id=row.row_id,
    )


def main() -> None:
    install_loguru_intercept()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a JSON config.")
    parser.add_argument("--output-dir", required=True, help="Where to write outputs.")
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Plumbing-only smoke run: 1 iter / 2 samples / 4 training rows.",
    )
    args = parser.parse_args()

    config = TrainingRunConfig.model_validate_json(Path(args.config).read_text())
    if args.debug:
        config = apply_debug_overrides(config)
        logger.info(
            "debug mode: num_iters={}, group_size={}, max_tokens={}",
            config.num_iters,
            config.group_size,
            config.max_tokens,
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_config.json").write_text(config.model_dump_json(indent=2))

    rows = load_rl_training_pool(packs=config.training_packs)
    logger.info("loaded {} training rows from HuggingFace", len(rows))
    if args.debug:
        rows = rows[:DEBUG_MAX_TRAINING_ROWS]
        logger.info("debug mode: clamped to {} rows", len(rows))

    items = [_row_to_item(r) for r in rows]
    artifact = train(config=config, items=items, output_dir=output_dir)
    artifact_path = output_dir / "training_artifact.json"
    artifact_path.write_text(artifact.model_dump_json(indent=2))
    logger.info(
        "training artifact at {} (checkpoint={})",
        artifact_path,
        artifact.checkpoint_uri,
    )


if __name__ == "__main__":
    main()
