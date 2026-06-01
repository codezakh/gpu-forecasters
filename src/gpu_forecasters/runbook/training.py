"""GRPO training for the three published surrogate variants.

Ports the trainer + reward functions that produced the e0158 /
e0159 / e0160 LoRA adapters into the public library so the runbook's
``02_train_surrogate.py`` script can reproduce them from a clean
checkout. The three variants differ only in the reward function — the
trainer and dataset builder are shared.

Reward functions:

* ``correctness``     — ``r_total = 1[predicted_bin == true_bin]``.
* ``correctness_brier`` — ``correctness + (1 - brier(p, y))``.
* ``correctness_crps``  — ``correctness + (1 - crps(p, y))``.

A parse failure (``estimate is None``) collapses every component to 0
under all three rewards.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import chz
from pydantic import BaseModel
from tinker_cookbook import checkpoint_utils, renderers
from tinker_cookbook.rl import train as rl_train
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    Metrics,
    RLDataset,
    RLDatasetBuilder,
    Trajectory,
)
from tinker_cookbook.tokenizer_utils import get_tokenizer

from gpu_forecasters.calibration.v2.scoring_rules import brier, crps
from gpu_forecasters.landscape_map.v2 import (
    KernelBinPredictionEnv,
    KernelRuntimeEstimate,
    LabeledKernelItem,
    RewardComponents,
    RewardFunction,
    SpeedupBin,
)
from gpu_forecasters.runbook.configs import (
    CorrectnessBrierRewardConfig,
    CorrectnessCrpsRewardConfig,
    CorrectnessRewardConfig,
    RewardConfig,
    TrainingRunConfig,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Reward functions
# ---------------------------------------------------------------------------


def _distance_diagnostic(
    estimate: KernelRuntimeEstimate | None, true_bin: SpeedupBin
) -> float:
    if estimate is None:
        return 0.0
    distance = abs(int(estimate.predicted_bin) - int(true_bin))
    return 1.0 - distance / 8.0


def _correctness(estimate: KernelRuntimeEstimate, true_bin: SpeedupBin) -> float:
    return 1.0 if estimate.predicted_bin == true_bin else 0.0


class _CorrectnessReward(RewardFunction):
    def reward(
        self, estimate: KernelRuntimeEstimate | None, true_bin: SpeedupBin
    ) -> RewardComponents:
        if estimate is None:
            return RewardComponents(distance=0.0, calibration=0.0, total=0.0)
        return RewardComponents(
            distance=_distance_diagnostic(estimate, true_bin),
            calibration=0.0,
            total=_correctness(estimate, true_bin),
        )


class _CorrectnessBrierReward(RewardFunction):
    def reward(
        self, estimate: KernelRuntimeEstimate | None, true_bin: SpeedupBin
    ) -> RewardComponents:
        if estimate is None:
            return RewardComponents(distance=0.0, calibration=0.0, total=0.0)
        calibration = 1.0 - brier(estimate.bin_probabilities, true_bin)
        correctness = _correctness(estimate, true_bin)
        return RewardComponents(
            distance=_distance_diagnostic(estimate, true_bin),
            calibration=calibration,
            total=correctness + calibration,
        )


class _CorrectnessCrpsReward(RewardFunction):
    def reward(
        self, estimate: KernelRuntimeEstimate | None, true_bin: SpeedupBin
    ) -> RewardComponents:
        if estimate is None:
            return RewardComponents(distance=0.0, calibration=0.0, total=0.0)
        calibration = 1.0 - crps(estimate.bin_probabilities, true_bin)
        correctness = _correctness(estimate, true_bin)
        return RewardComponents(
            distance=_distance_diagnostic(estimate, true_bin),
            calibration=calibration,
            total=correctness + calibration,
        )


def build_reward_function(config: RewardConfig) -> RewardFunction:
    """Construct the reward instance the GRPO env will call per step."""
    if isinstance(config, CorrectnessRewardConfig):
        return _CorrectnessReward()
    if isinstance(config, CorrectnessBrierRewardConfig):
        return _CorrectnessBrierReward()
    if isinstance(config, CorrectnessCrpsRewardConfig):
        return _CorrectnessCrpsReward()
    raise TypeError(f"unknown reward config: {config!r}")


# ---------------------------------------------------------------------------
# Dataset builder
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _LabeledKernelGroupBuilder(EnvGroupBuilder):
    item: LabeledKernelItem
    model_name_for_tokenizer: str
    renderer_name: str
    reward_config: RewardConfig
    group_size: int

    async def make_envs(self) -> Sequence[Env]:
        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        renderer = renderers.get_renderer(self.renderer_name, tokenizer=tokenizer)
        reward_fn = build_reward_function(self.reward_config)
        return [
            KernelBinPredictionEnv(self.item, renderer, reward_fn)
            for _ in range(self.group_size)
        ]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        del env_group
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> list[str]:
        return [f"v2_kernel_bin_prediction_{self.reward_config.kind}", self.item.pack_name]


class _LabeledKernelDataset(RLDataset):
    def __init__(
        self, groups: list[_LabeledKernelGroupBuilder], groups_per_batch: int
    ) -> None:
        self._groups = groups
        self._groups_per_batch = groups_per_batch

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        n = len(self._groups)
        start = (index * self._groups_per_batch) % n
        end = start + self._groups_per_batch
        if end <= n:
            return self._groups[start:end]
        return list(self._groups[start:n]) + list(self._groups[: end - n])

    def __len__(self) -> int:
        return max(1, math.ceil(len(self._groups) / self._groups_per_batch))


@chz.chz
class _LabeledKernelDatasetBuilder(RLDatasetBuilder):
    items: tuple[LabeledKernelItem, ...]
    model_name_for_tokenizer: str
    renderer_name: str
    reward_config: RewardConfig
    group_size: int
    groups_per_batch: int

    async def __call__(self) -> tuple[_LabeledKernelDataset, None]:
        groups = [
            _LabeledKernelGroupBuilder(
                item=item,
                model_name_for_tokenizer=self.model_name_for_tokenizer,
                renderer_name=self.renderer_name,
                reward_config=self.reward_config,
                group_size=self.group_size,
            )
            for item in self.items
        ]
        return (
            _LabeledKernelDataset(groups=groups, groups_per_batch=self.groups_per_batch),
            None,
        )


# ---------------------------------------------------------------------------
# Trainer + artifact
# ---------------------------------------------------------------------------


class TrainingTokenUsage(BaseModel, frozen=True):
    sampling_input_tokens: int
    sampling_output_tokens: int
    num_examples: int
    num_steps: int


class TrainingArtifact(BaseModel, frozen=True):
    """End-of-run record of one training run.

    The Tinker ``checkpoint_uri`` is the load-bearing field — it's what
    a downstream reader passes to the SamplingClient to use the trained
    adapter. The other fields are provenance for the matching dataset
    card.
    """

    base_model: str
    renderer_name: str
    checkpoint_uri: str
    reward_kind: str
    token_usage: TrainingTokenUsage
    trainer_config_snapshot: dict[str, Any]


def _harvest_token_usage(
    log_path: Path, num_examples: int, num_steps: int
) -> TrainingTokenUsage:
    """Sum action / observation tokens from the cookbook's ``metrics.jsonl``."""
    metrics_path = log_path / "metrics.jsonl"
    if not metrics_path.exists():
        return TrainingTokenUsage(
            sampling_input_tokens=0,
            sampling_output_tokens=0,
            num_examples=num_examples,
            num_steps=num_steps,
        )
    total_ac = 0
    total_ob = 0
    with metrics_path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            ac = row.get("env/all/total_ac_tokens", row.get("total_ac_tokens", 0))
            ob = row.get("env/all/total_ob_tokens", row.get("total_ob_tokens", 0))
            if isinstance(ac, (int, float)):
                total_ac += int(ac)
            if isinstance(ob, (int, float)):
                total_ob += int(ob)
    return TrainingTokenUsage(
        sampling_input_tokens=total_ob,
        sampling_output_tokens=total_ac,
        num_examples=num_examples,
        num_steps=num_steps,
    )


def train(
    *, config: TrainingRunConfig, items: list[LabeledKernelItem], output_dir: Path
) -> TrainingArtifact:
    """Run one GRPO training pass over ``items`` into ``output_dir``.

    Writes the cookbook's training logs under ``<output_dir>/training/``
    and returns a ``TrainingArtifact`` whose ``checkpoint_uri`` is the
    Tinker URI of the final sampler-weights checkpoint.
    """
    log_path = output_dir / "training"
    log_path.mkdir(parents=True, exist_ok=True)

    dataset_builder = _LabeledKernelDatasetBuilder(
        items=tuple(items),
        model_name_for_tokenizer=config.base_model,
        renderer_name=config.renderer_name,
        reward_config=config.reward,
        group_size=config.group_size,
        groups_per_batch=config.groups_per_batch,
    )

    rl_config = rl_train.Config(
        learning_rate=config.learning_rate,
        dataset_builder=dataset_builder,
        model_name=config.base_model,
        max_tokens=config.max_tokens,
        log_path=str(log_path),
        renderer_name=config.renderer_name,
        lora_rank=config.lora_rank,
        temperature=config.temperature,
        max_steps=config.num_iters,
        save_every=config.save_every,
        eval_every=0,
        wandb_project=None,
        wandb_name=None,
        remove_constant_reward_groups=True,
    )

    logger.info(
        "GRPO training: %d examples, %d iters, group_size=%d, "
        "groups_per_batch=%d, save_every=%d, max_tokens=%d, reward=%s",
        len(items),
        config.num_iters,
        config.group_size,
        config.groups_per_batch,
        config.save_every,
        config.max_tokens,
        config.reward.kind,
    )
    asyncio.run(rl_train.main(rl_config))

    record = checkpoint_utils.get_last_checkpoint(
        str(log_path), required_key="sampler_path"
    )
    if record is None or record.sampler_path is None:
        raise RuntimeError(
            f"No sampler-weight checkpoint found in {log_path}; "
            "training may have failed before the first save."
        )
    checkpoint_uri = record.sampler_path

    token_usage = _harvest_token_usage(
        log_path, num_examples=len(items), num_steps=config.num_iters
    )

    return TrainingArtifact(
        base_model=config.base_model,
        renderer_name=config.renderer_name,
        checkpoint_uri=checkpoint_uri,
        reward_kind=config.reward.kind,
        token_usage=token_usage,
        trainer_config_snapshot=json.loads(config.model_dump_json()),
    )


__all__ = [
    "TrainingArtifact",
    "TrainingTokenUsage",
    "build_reward_function",
    "train",
]
