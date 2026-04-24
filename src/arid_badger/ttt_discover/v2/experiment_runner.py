"""Reusable multi-seed experiment driver + config snapshot types for v2.

Every v2 experiment follows the same shape:

1. Declare an ``ExperimentConfig`` — a frozen Pydantic record of
   ``num_runs`` plus a ``DiscoverConfigSnapshot`` of the hparams that
   would otherwise be split across ``chz``-decorated ``DiscoverConfig``
   instances (Pydantic is serialisable to ``experiment.json`` for
   reproducibility; ``chz`` is not).
2. Call ``run_experiment(output_dir, config)`` to persist the config and
   iterate the configured number of seeds, each invoking
   ``ttt_discover.v2.discover`` with ``experiment_name = "seed_{NN}"`` —
   the log layout every v2 experiment shares.

``run_experiment`` is the second (non-smoke) importer's way of not
copy-pasting the driver out of an experiment; ``print_progress`` is the
paired formatter ``results.py`` usually wants.
"""

from __future__ import annotations

import os
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from arid_badger.ttt_discover.v2.discovery import DiscoverConfig, discover
from arid_badger.ttt_discover.v2.results import V2ExperimentResults


class DiscoverConfigSnapshot(BaseModel, frozen=True):
    """Serialisable snapshot of the ``DiscoverConfig`` knobs an
    experiment cares about. Translated to ``DiscoverConfig`` at seed
    launch time; ``chz`` configs aren't JSON-roundtrippable so we keep
    this shape separate."""

    model_name: str
    renderer_name: str
    group_size: int
    groups_per_batch: int
    num_epochs: int
    lora_rank: int
    learning_rate: float
    temperature: float
    kl_penalty_coef: float
    phase1_max_tokens: int
    save_every: int


class ExperimentConfig(BaseModel, frozen=True):
    num_runs: int
    discover: DiscoverConfigSnapshot


def _build_discover_config(
    snapshot: DiscoverConfigSnapshot, experiment_name: str
) -> DiscoverConfig:
    return DiscoverConfig(
        model_name=snapshot.model_name,
        renderer_name=snapshot.renderer_name,
        group_size=snapshot.group_size,
        groups_per_batch=snapshot.groups_per_batch,
        num_epochs=snapshot.num_epochs,
        save_every=snapshot.save_every,
        lora_rank=snapshot.lora_rank,
        learning_rate=snapshot.learning_rate,
        temperature=snapshot.temperature,
        kl_penalty_coef=snapshot.kl_penalty_coef,
        phase1_max_tokens=snapshot.phase1_max_tokens,
        experiment_name=experiment_name,
        wandb_project=None,
    )


def _seed_log_dir(workspace_dir: Path, experiment_name: str) -> Path:
    return workspace_dir / "tinker_log" / experiment_name


def _write_experiment_config(output_dir: Path, config: ExperimentConfig) -> None:
    path = output_dir / "experiment.json"
    tmp = path.with_name(path.name + ".tmp")
    _ = tmp.write_text(config.model_dump_json(indent=2))
    _ = tmp.replace(path)


def run_experiment(output_dir: Path, config: ExperimentConfig) -> None:
    """Run ``config.num_runs`` seeds under ``output_dir``.

    Each seed writes ``tinker_log/seed_NN/rollouts.jsonl`` (plus PUCT
    snapshots) and a ``DONE`` marker on completion; re-running skips
    seeds whose marker exists, so runs are resumable at seed
    granularity.

    Note: ``discover(...)`` resolves its log path relative to cwd (see
    ``discovery.py::_resolve_paths``), so this driver ``os.chdir``s into
    ``output_dir``. Callers who care about cwd should resolve
    ``output_dir`` before invoking.
    """
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_experiment_config(output_dir, config)

    os.chdir(output_dir)

    logger.info(
        "Experiment starting: model={model} num_runs={n} output_dir={dir}",
        model=config.discover.model_name,
        n=config.num_runs,
        dir=output_dir,
    )

    for i in range(config.num_runs):
        experiment_name = f"seed_{i:02d}"
        seed_dir = _seed_log_dir(output_dir, experiment_name)
        done_marker = seed_dir / "DONE"
        if done_marker.exists():
            logger.info("Skipping seed {i}: {path} exists", i=i, path=done_marker)
            continue

        logger.info("Seed {i} starting: log_dir={dir}", i=i, dir=seed_dir)
        discover(_build_discover_config(config.discover, experiment_name))

        seed_dir.mkdir(parents=True, exist_ok=True)
        done_marker.touch()
        logger.info("Seed {i} done: {dir}", i=i, dir=seed_dir)

    logger.info("Experiment done: {dir}", dir=output_dir)


def print_progress(results: V2ExperimentResults) -> None:
    """One-line-per-seed formatter for ``V2ExperimentResults``. Stable
    shape; safe to parse for dashboards."""
    seeds = results.per_seed()
    if not seeds:
        print(f"no seeds yet under {results.output_dir}")
        return
    for seed in seeds:
        summary = seed.summary()
        bests = seed.best_by_step()
        latest_best = bests[-1].best_reward if bests else None
        print(
            f"seed {seed.seed_index:02d}: "
            f"rollouts={summary.num_rollouts} "
            f"success={summary.num_successes} "
            f"parse_fail={summary.num_parse_failures} "
            f"compile_fail={summary.num_compile_failures} "
            f"runtime_err={summary.num_runtime_errors} "
            f"incorrect={summary.num_incorrect} "
            f"infra_fail={summary.num_infra_failures} "
            f"best_reward={latest_best}"
        )
