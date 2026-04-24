"""Top-level entry point for a v2 TTT-Discover run.

Mirrors v1's ``DiscoverConfig`` / ``discover`` surface so experiments
look nearly identical to the e006x series. Build the components via
``build_default_components`` (or supply your own) and call ``discover``.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from pathlib import Path

import chz

from arid_badger.hill_climbing.mutation_providers.trimul_feedback_mutation import (
    _TRIMUL_BASE_PROMPT,
)
from arid_badger.trimul.cases import BENCHMARK_CASES
from arid_badger.ttt_discover.v1.rl.train import Config as V1TrainConfig
from arid_badger.ttt_discover.v1.rl.train import main as v1_train_main
from arid_badger.ttt_discover.v1.tinker_utils import misc_utils as v1_misc_utils
from arid_badger.ttt_discover.v2.archive.puct import PUCTCandidateArchive
from arid_badger.ttt_discover.v2.domain.problem import TriMulProblem
from arid_badger.ttt_discover.v2.evaluator.modal_trimul import ModalTriMulEvaluator
from arid_badger.ttt_discover.v2.extractors.python_block import (
    LastPythonBlockExtractor,
)
from arid_badger.ttt_discover.v2.renderers.feedback_trimul import (
    TriMulFeedbackPromptRenderer,
)
from arid_badger.ttt_discover.v2.renderers.task_static import (
    StaticTaskPromptRenderer,
)
from arid_badger.ttt_discover.v2.rl_integration import (
    V2Components,
    V2RLDatasetBuilder,
)
from arid_badger.ttt_discover.v2.scalarizers.by_target_us import ScaleByTargetUs
from arid_badger.ttt_discover.v2.sinks.jsonl import JsonlRolloutSink

logger = logging.getLogger(__name__)


DEFAULT_GPU_NAME = "A100-80GB"
DEFAULT_TRITON_VERSION = "3.3.1"
# A100 paper-SOTA TriMul geomean runtime; used as the reward scalarizer's
# numerator so that a kernel matching paper-SOTA gets ~1.0 reward.
DEFAULT_TARGET_RUNTIME_US = 2500.0


def build_default_problem(
    *,
    gpu_name: str = DEFAULT_GPU_NAME,
    triton_version: str = DEFAULT_TRITON_VERSION,
    target_runtime_us: float = DEFAULT_TARGET_RUNTIME_US,
) -> TriMulProblem:
    return TriMulProblem(
        base_prompt_text=_TRIMUL_BASE_PROMPT,
        test_cases=tuple(BENCHMARK_CASES),
        gpu_name=gpu_name,
        triton_version=triton_version,
        target_runtime_us=target_runtime_us,
    )


def build_default_components(
    *,
    log_path: Path,
    problem: TriMulProblem | None = None,
) -> V2Components:
    """Factory for the default v2 component wiring.

    The archive snapshot directory is ``log_path``; the event log file is
    ``log_path/rollouts.jsonl``. Both are created lazily on first write.
    """
    problem = problem or build_default_problem()
    archive = PUCTCandidateArchive(directory=log_path)
    sink = JsonlRolloutSink(path=log_path / "rollouts.jsonl")
    evaluator = ModalTriMulEvaluator(
        gpu_name=problem.gpu_name,
        test_cases=list(problem.test_cases),
    )
    return V2Components(
        problem=problem,
        task_prompt_renderer=StaticTaskPromptRenderer(),
        feedback_prompt_renderer=TriMulFeedbackPromptRenderer(),
        evaluator=evaluator,
        scalarizer=ScaleByTargetUs(target_us=problem.target_runtime_us),
        extractor=LastPythonBlockExtractor(),
        archive=archive,
        sink=sink,
    )


@chz.chz
class DiscoverConfig:
    """Subset of v1's ``DiscoverConfig`` that v2 actually uses.

    v2 drops ``env_type`` / ``problem_type`` / ``num_cpus_per_task``
    (those were v1's cross-env dispatch knobs) and adds no new fields —
    the TriMul-specific configuration lives on ``TriMulProblem`` and can
    be overridden via ``problem`` when ``components`` is built manually.
    """

    # Model / training
    model_name: str = "openai/gpt-oss-120b"
    lora_rank: int = 32
    renderer_name: str = "gpt_oss_medium_reasoning"
    save_every: int = 1

    group_size: int = 4
    groups_per_batch: int = 2
    learning_rate: float = 4e-5
    num_epochs: int = 40
    temperature: float = 1.0
    kl_penalty_coef: float = 0.1
    phase1_max_tokens: int = 16000

    # Experiment metadata
    experiment_name: str = "ttt-discover-v2"
    wandb_project: str | None = None


@dataclass(frozen=True)
class _ResolvedPaths:
    log_path: Path
    log_file: Path


def _resolve_paths(experiment_name: str) -> _ResolvedPaths:
    # Mirror v1: ``./tinker_log/{experiment_name}/`` relative to cwd.
    log_path = Path(f"./tinker_log/{experiment_name}").resolve()
    log_file = log_path / "train.log"
    return _ResolvedPaths(log_path=log_path, log_file=log_file)


async def _discover_impl(cfg: DiscoverConfig) -> None:
    assert cfg.model_name in {
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b",
    }, "Only supporting GPT-OSS models for now."

    paths = _resolve_paths(cfg.experiment_name)
    paths.log_path.mkdir(parents=True, exist_ok=True)

    components = build_default_components(log_path=paths.log_path)

    dataset_builder = V2RLDatasetBuilder(
        components=components,
        model_name_for_tokenizer=cfg.model_name,
        renderer_name=cfg.renderer_name,
        groups_per_batch=cfg.groups_per_batch,
        group_size=cfg.group_size,
    )

    # v1's ``Config.env_type`` is annotated ``type`` and the training
    # loop only reads it via ``getattr(env_type, 'env_name', ...)`` for
    # logging; a sentinel class with a name satisfies that.
    class _V2Sentinel:
        env_name = "trimul_v2"

    rl_config = V1TrainConfig(
        env_type=_V2Sentinel,
        problem_type="trimul",
        learning_rate=cfg.learning_rate,
        dataset_builder=dataset_builder,
        model_name=cfg.model_name,
        lora_rank=cfg.lora_rank,
        temperature=cfg.temperature,
        wandb_project=cfg.wandb_project,
        wandb_name=cfg.experiment_name,
        log_path=str(paths.log_path),
        load_checkpoint_path=None,
        kl_penalty_coef=cfg.kl_penalty_coef,
        num_substeps=1,
        save_every=cfg.save_every,
        num_epochs=cfg.num_epochs,
        loss_fn="importance_sampling",
        adv_estimator="entropic_adaptive_beta",
        adv_estimator_beta=2.0,
        remove_constant_reward_groups=True,
        phase1_max_tokens=cfg.phase1_max_tokens,
        local_model_path=None,
    )

    v1_misc_utils.check_log_dir(str(paths.log_path), behavior_if_exists="resume")
    os.makedirs(paths.log_path, exist_ok=True)
    logging.getLogger().handlers.clear()
    logging.getLogger().addHandler(logging.NullHandler())
    logging.basicConfig(
        level=logging.INFO, filename=str(paths.log_file), filemode="a", force=True
    )
    logger.info("Logging to %s", paths.log_file)

    await v1_train_main(rl_config)


def discover(cfg: DiscoverConfig) -> None:
    asyncio.run(_discover_impl(cfg))
