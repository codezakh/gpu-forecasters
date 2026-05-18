"""Pack-bound entry point for v3 surrogate-filtered PUCT searches.

Mirrors ``gpu_mode_kernel.experiment_helper.run_pack_experiment`` for
the v3 surrogate flow. Per-pack experiment files become ~25 lines:
import the pack runtime + case-speedup type, instantiate one
``SurrogateSearchExperimentConfig``, call ``run_pack_experiment``.

Idempotency follows the v2 helper: each run lands in ``run_NN/`` with
a ``summary.json`` written last. Re-invoking after a crash skips
completed runs and re-enters partial ones via the v3 driver's
replay-from-log recovery path.
"""

from __future__ import annotations

import time
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from arid_badger.gpu_mode_kernel.core import (
    CaseSpeedupT,
    GpuModeKernelObservation,
)
from arid_badger.gpu_mode_kernel.kernel_pack import TestArgsT
from arid_badger.gpu_mode_kernel.modal_scoring import PackedModalRuntime
from arid_badger.gpu_mode_kernel.providers.v2_feedback_mutation import (
    GpuModeKernelFeedbackMutationProvider,
)
from arid_badger.gpu_mode_kernel.providers.v2_modal_scoring import (
    GpuModeKernelModalProvider,
)
from arid_badger.gpu_mode_kernel.surrogate_search.v1.config import (
    LiteLlmSurrogateConfig,
    SurrogateConfig,
    SurrogateSearchExperimentConfig,
    TinkerSurrogateConfig,
)
from arid_badger.landscape_map.v2 import (
    KernelTaskInfo,
    RetryingSpeedupEstimator,
    TinkerSamplingClientEstimator,
)
from arid_badger.landscape_map.v2.litellm_estimator import LlmSpeedupEstimator
from arid_badger.max_reward_puct.v3.event_log import FileEventLog
from arid_badger.max_reward_puct.v3.events import (
    CandidateDeferred,
    CandidateSelected,
    EvaluationCompleted,
    EvaluationFailed,
    ForecastCompleted,
    ForecastFailed,
    MutationFailed,
    SearchInitialized,
    StepCompleted,
)
from arid_badger.max_reward_puct.v3.providers import SpeedupEstimator
from arid_badger.max_reward_puct.v3.scoring_providers import (
    CoroutineSpeedupEstimator,
)
from arid_badger.max_reward_puct.v3.search import SearchDriver
from arid_badger.max_reward_puct.v3.state import replay


class RunSummary(BaseModel, frozen=True):
    """End-of-run snapshot derived from the event log.

    ``best_reward`` is None when no archived node has a reward (no
    correct kernel produced). ``num_forecast_*`` and ``num_deferred``
    capture the surrogate's behaviour; ``num_evaluations_total`` is
    the paid-GPU count that matters for budget-matched comparisons.
    """

    steps_completed: int
    archive_size: int
    best_reward: float | None
    seed_reward: float | None
    best_node_ulid: str | None
    num_evaluations_total: int
    num_evaluations_correct: int
    num_evaluation_failures: int
    num_mutation_failures: int
    num_forecasts_completed: int
    num_forecasts_failed: int
    num_deferred: int
    num_selected: int
    wall_clock_seconds: float


def _atomic_write_json(model: BaseModel, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    _ = tmp.write_text(model.model_dump_json(indent=2))
    _ = tmp.replace(path)


def _build_surrogate(config: SurrogateConfig) -> SpeedupEstimator:
    """Construct the surrogate's outer ``SpeedupEstimator`` from config.

    Wraps the variant-specific inner estimator in a
    ``RetryingSpeedupEstimator`` (parse-error retries) and a
    ``CoroutineSpeedupEstimator`` (async-to-Future adapter for the v3
    driver). The returned object is a context manager whose lifetime
    the caller owns.
    """
    match config:
        case TinkerSurrogateConfig():
            inner = TinkerSamplingClientEstimator(
                base_model=config.base_model,
                model_path=config.checkpoint_uri,
                renderer_name=config.renderer_name,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
            )
        case LiteLlmSurrogateConfig():
            inner = LlmSpeedupEstimator(
                model_slug=config.model_slug,
                temperature=config.temperature,
                max_tokens=config.max_tokens,
                request_timeout_s=config.request_timeout_s,
            )
    retrying = RetryingSpeedupEstimator(inner, max_retries=config.max_retries)
    return CoroutineSpeedupEstimator(retrying)


def _compute_summary(
    event_log: FileEventLog[GpuModeKernelObservation[CaseSpeedupT]],
    config: SurrogateSearchExperimentConfig,
    wall_clock_seconds: float,
    observation_type: type[GpuModeKernelObservation[CaseSpeedupT]],
) -> RunSummary:
    events = event_log.read_all()
    state = replay(
        events,
        k_per_parent=config.search.k_per_parent,
        archive_capacity=config.search.archive_capacity,
        observation_type=observation_type,
    )

    num_eval_completed = 0
    num_eval_correct = 0
    num_eval_failed = 0
    num_mut_failed = 0
    num_fc_completed = 0
    num_fc_failed = 0
    num_deferred = 0
    num_selected = 0
    num_steps = 0
    seed_reward: float | None = None
    for event in events:
        if isinstance(event, SearchInitialized):
            seed_reward = event.root.evaluation.reward
        elif isinstance(event, EvaluationCompleted):
            num_eval_completed += 1
            if event.evaluation.reward is not None:
                num_eval_correct += 1
        elif isinstance(event, EvaluationFailed):
            num_eval_failed += 1
        elif isinstance(event, MutationFailed):
            num_mut_failed += 1
        elif isinstance(event, ForecastCompleted):
            num_fc_completed += 1
        elif isinstance(event, ForecastFailed):
            num_fc_failed += 1
        elif isinstance(event, CandidateSelected):
            num_selected += 1
        elif isinstance(event, CandidateDeferred):
            num_deferred += 1
        elif isinstance(event, StepCompleted):
            num_steps += 1

    best = state.best_archived_node()
    return RunSummary(
        steps_completed=num_steps,
        archive_size=len(state.archive),
        best_reward=None if best is None else best.evaluation.reward,
        seed_reward=seed_reward,
        best_node_ulid=None if best is None else str(best.ulid),
        num_evaluations_total=num_eval_completed,
        num_evaluations_correct=num_eval_correct,
        num_evaluation_failures=num_eval_failed,
        num_mutation_failures=num_mut_failed,
        num_forecasts_completed=num_fc_completed,
        num_forecasts_failed=num_fc_failed,
        num_deferred=num_deferred,
        num_selected=num_selected,
        wall_clock_seconds=wall_clock_seconds,
    )


def _run_single(
    *,
    pack_runtime: PackedModalRuntime[TestArgsT, CaseSpeedupT],
    observation_type: type[GpuModeKernelObservation[CaseSpeedupT]],
    run_dir: Path,
    config: SurrogateSearchExperimentConfig,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "events.jsonl"

    event_log: FileEventLog[GpuModeKernelObservation[CaseSpeedupT]] = FileEventLog(
        log_path, observation_type=observation_type
    )

    pack_name = pack_runtime.pack.name
    # The surrogate prompt treats KernelTaskInfo as opaque metadata —
    # what matters is that op_name reflects the pack identity.
    kernel_task = KernelTaskInfo(op_name=pack_name, level_id=0, task_id=0)

    mutation_provider = GpuModeKernelFeedbackMutationProvider(
        pack=pack_runtime.pack,
        model_slug=config.mutator.model_slug,
        gpu_name=config.evaluation.gpu,
        max_llm_concurrency=config.mutator.max_llm_concurrency,
        num_retries=config.mutator.num_retries,
        request_timeout_s=config.mutator.request_timeout_s,
        temperature=config.mutator.temperature,
        max_tokens=config.mutator.max_tokens,
    )
    evaluation_provider = GpuModeKernelModalProvider(
        pack_runtime=pack_runtime,
        aggregator=config.evaluation.aggregator,
        gpu=config.evaluation.gpu,
        max_in_flight=config.evaluation.max_in_flight,
        invocation_sink=None,
    )
    surrogate = _build_surrogate(config.surrogate)

    logger.info(
        "v3 {pack} run starting in {dir}: steps={steps} batch={batch} spp={spp} k={k} mutator={mut} surrogate={sk}",
        pack=pack_name,
        dir=run_dir,
        steps=config.search.total_budget_steps,
        batch=config.search.batch_size,
        spp=config.search.samples_per_parent,
        k=config.search.k_per_parent,
        mut=config.mutator.model_slug,
        sk=config.surrogate.kind,
    )

    start_s = time.perf_counter()
    with mutation_provider, evaluation_provider, surrogate as surrogate_handle:
        driver = SearchDriver[GpuModeKernelObservation[CaseSpeedupT]](
            config.search,
            mutation_provider=mutation_provider,
            evaluation_provider=evaluation_provider,
            surrogate=surrogate_handle,
            kernel_task=kernel_task,
            seed_reference_code=pack_runtime.pack.seed_kernel_code,
            hardware=config.hardware,
            event_log=event_log,
            observation_type=observation_type,
        )
        _ = driver.run(initial_program=pack_runtime.pack.seed_kernel_code)
    wall_clock_seconds = time.perf_counter() - start_s

    summary = _compute_summary(
        event_log, config, wall_clock_seconds, observation_type=observation_type
    )
    _atomic_write_json(summary, run_dir / "summary.json")

    best_str = (
        f"{summary.best_reward:.4f}x" if summary.best_reward is not None else "none"
    )
    logger.info(
        "v3 {pack} run done in {dir}: steps={steps} archive={size} best={best} evals={c}/{n} forecasts={fc} deferred={d} wall={wall:.1f}s",
        pack=pack_name,
        dir=run_dir,
        steps=summary.steps_completed,
        size=summary.archive_size,
        best=best_str,
        c=summary.num_evaluations_correct,
        n=summary.num_evaluations_total,
        fc=summary.num_forecasts_completed,
        d=summary.num_deferred,
        wall=wall_clock_seconds,
    )


def run_pack_experiment(
    *,
    pack_runtime: PackedModalRuntime[TestArgsT, CaseSpeedupT],
    case_speedup_type: type[CaseSpeedupT],
    output_dir: Path,
    config: SurrogateSearchExperimentConfig,
) -> None:
    """Run ``config.num_runs`` v3 surrogate-filtered searches into ``output_dir``.

    The observation type is derived from ``case_speedup_type`` via
    Pydantic generic subscription — the same construction the per-pack
    experiment files use directly.

    Already-completed runs (those with a ``summary.json``) are
    skipped, making this safe to re-invoke after a crash. The v3
    driver's log-replay path resumes mid-step on partial runs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(config, output_dir / "experiment.json")

    observation_type: type[GpuModeKernelObservation[CaseSpeedupT]] = (
        GpuModeKernelObservation[case_speedup_type]
    )

    pack_name = pack_runtime.pack.name
    logger.info(
        "v3 {pack} experiment starting: mutator={mut} surrogate={sk} num_runs={n} output_dir={dir}",
        pack=pack_name,
        mut=config.mutator.model_slug,
        sk=config.surrogate.kind,
        n=config.num_runs,
        dir=output_dir,
    )

    for i in range(config.num_runs):
        run_dir = output_dir / f"run_{i:02d}"
        if (run_dir / "summary.json").exists():
            logger.info("Skipping run {i}: summary.json already present", i=i)
            continue
        _run_single(
            pack_runtime=pack_runtime,
            observation_type=observation_type,
            run_dir=run_dir,
            config=config,
        )

    logger.info("v3 {pack} experiment done: {dir}", pack=pack_name, dir=output_dir)


def load_run_summaries(root: Path) -> list[RunSummary]:
    """Read every ``run_*/summary.json`` under ``root`` into ``RunSummary``s.

    Returns ``[]`` if the directory does not exist yet.
    """
    if not root.exists():
        return []
    return [
        RunSummary.model_validate_json(p.read_text())
        for p in sorted(root.glob("run_*/summary.json"))
    ]


__all__ = [
    "RunSummary",
    "load_run_summaries",
    "run_pack_experiment",
]
