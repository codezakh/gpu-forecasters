"""Generic experiment harness for compound (search-embedded abstention)
PUCT runs.

Lifted from the duplicated wiring in ``e0155`` and ``e0156``: both
files compose a ``CompoundEvaluationProvider`` + a
``CompoundFeedbackMutationProvider`` around the same Modal evaluator,
run a v2 ``SearchDriver`` with a ``CompoundObservation`` event log,
and produce a ``CompoundRunSummary`` derived from the log. Per the
library-evolution guidance, two consumers of an identical pattern is
the trigger to promote.

Each compound experiment now collapses to ~30 lines: imports plus a
``CONFIG = CompoundExperimentConfig(...)`` instantiation, with
``run_compound_pack_experiment`` doing the rest.

Resumability: identical to the always-real harness — the v2 driver
itself is resumable, and runs whose ``summary.json`` is already on
disk are skipped entirely.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import cast

from loguru import logger
from pydantic import BaseModel

from arid_badger.abstaining_evaluation.v1.forecast_reward import (
    ExpectedSpeedupReward,
)
from arid_badger.abstaining_evaluation.v1.mutation_provider import (
    CompoundFeedbackMutationProvider,
)
from arid_badger.abstaining_evaluation.v1.observation import (
    CompoundObservation,
    ForecastObservation,
    RealObservation,
)
from arid_badger.abstaining_evaluation.v1.provider import (
    CompoundEvaluationProvider,
)
from arid_badger.gpu_mode_kernel.aggregation import AggregationMethod
from arid_badger.gpu_mode_kernel.core import (
    CaseSpeedupT,
    SuccessFeedback,
)
from arid_badger.gpu_mode_kernel.kernel_pack import TestArgsT
from arid_badger.gpu_mode_kernel.modal_scoring import PackedModalRuntime
from arid_badger.gpu_mode_kernel.providers.v2_modal_scoring import (
    GpuModeKernelModalProvider,
)
from arid_badger.landscape_map.v2 import (
    HardwareContext,
    KernelImplementation,
    KernelTaskInfo,
)
from arid_badger.landscape_map.v2.abstain_estimator import (
    AbstainingLlmSpeedupEstimator,
)
from arid_badger.max_reward_puct.v2.config import SearchConfig
from arid_badger.max_reward_puct.v2.event_log import FileEventLog
from arid_badger.max_reward_puct.v2.events import (
    EvaluationCompleted,
    SearchInitialized,
)
from arid_badger.max_reward_puct.v2.search import SearchDriver
from arid_badger.max_reward_puct.v2.state import replay


# ---------------------------------------------------------------------------
# Config schemas
# ---------------------------------------------------------------------------


class LlmConfig(BaseModel, frozen=True):
    """Knobs that bind to a single LiteLLM-hosted model.

    Used twice in a compound config — once for the abstaining
    surrogate, once for the mutator. The two are intentionally
    separate so a future cell can swap one without the other (e.g.
    keep the gpt-oss-20b mutator we have a baseline for, but try a
    larger surrogate). ``max_tokens=None`` is the right setting for
    Gemini 3 Flash — it rejects an explicit cap. Together-hosted
    gpt-oss-20b requires a cap (e.g. 32000).
    """

    model_slug: str
    max_concurrency: int
    max_tokens: int | None
    request_timeout_s: float
    temperature: float = 1.0
    num_retries: int = 0


class CompoundProviderConfig(BaseModel, frozen=True):
    """Knobs that bind to the concrete v2 providers."""

    surrogate: LlmConfig
    mutator: LlmConfig
    real_evaluator_max_in_flight: int
    gpu: str
    aggregator: AggregationMethod
    hardware: HardwareContext


class CompoundRunConfig(BaseModel, frozen=True):
    """One compound search worth of config."""

    search: SearchConfig
    provider: CompoundProviderConfig


class CompoundExperimentConfig(BaseModel, frozen=True):
    """Top-level compound experiment config."""

    num_runs: int
    run: CompoundRunConfig


class CompoundRunSummary(BaseModel, frozen=True):
    """End-of-run snapshot derived from the compound event log.

    ``best_forecast_reward`` is the max scalar reward across every
    completed evaluation in the log — what the search "claimed" at
    the end of the run, including unconfirmed surrogate forecasts.
    ``best_confirmed_reward`` is the max over completions whose
    observation is a ``RealObservation`` carrying ``SuccessFeedback`` —
    what the GPU actually saw and timed. Either may be ``None``: the
    confirmed one if the surrogate never deferred to a successful
    real eval; the forecast one if no completion ever produced a
    reward (every candidate failed).
    """

    steps_completed: int
    archive_size: int
    n_forecasts: int
    n_real_evals: int
    n_deferrals: int
    n_evaluations_total: int
    n_evaluations_failed: int
    best_forecast_reward: float | None
    best_confirmed_reward: float | None
    seed_reward: float | None
    wall_clock_seconds: float


# ---------------------------------------------------------------------------
# Per-run summary computation (pure function of an event log on disk)
# ---------------------------------------------------------------------------


def _compute_summary(
    event_log: FileEventLog[CompoundObservation[CaseSpeedupT]],
    search_config: SearchConfig,
    wall_clock_seconds: float,
    observation_type: type[CompoundObservation[CaseSpeedupT]],
) -> CompoundRunSummary:
    events = event_log.read_all()
    state = replay(
        events,
        k_per_parent=search_config.k_per_parent,
        archive_capacity=search_config.archive_capacity,
        observation_type=observation_type,
    )

    n_forecasts = 0
    n_real_evals = 0
    n_deferrals = 0
    n_evals_total = 0
    n_evals_failed = 0
    best_forecast_reward: float | None = None
    best_confirmed_reward: float | None = None
    seed_reward: float | None = None

    for event in events:
        if isinstance(event, SearchInitialized):
            seed_reward = event.root.evaluation.reward
            continue
        if not isinstance(event, EvaluationCompleted):
            continue
        n_evals_total += 1
        evaluation = event.evaluation
        if evaluation.reward is None:
            n_evals_failed += 1
            continue
        if best_forecast_reward is None or evaluation.reward > best_forecast_reward:
            best_forecast_reward = evaluation.reward
        observation = evaluation.observation
        if isinstance(observation, ForecastObservation):
            n_forecasts += 1
        elif isinstance(observation, RealObservation):
            n_real_evals += 1
            if observation.deferral_reason is not None:
                n_deferrals += 1
            if isinstance(observation.inner.feedback, SuccessFeedback):
                if (
                    best_confirmed_reward is None
                    or evaluation.reward > best_confirmed_reward
                ):
                    best_confirmed_reward = evaluation.reward

    return CompoundRunSummary(
        steps_completed=state.current_step,
        archive_size=len(state.archive),
        n_forecasts=n_forecasts,
        n_real_evals=n_real_evals,
        n_deferrals=n_deferrals,
        n_evaluations_total=n_evals_total,
        n_evaluations_failed=n_evals_failed,
        best_forecast_reward=best_forecast_reward,
        best_confirmed_reward=best_confirmed_reward,
        seed_reward=seed_reward,
        wall_clock_seconds=wall_clock_seconds,
    )


def _atomic_write_json(model: BaseModel, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    _ = tmp.write_text(model.model_dump_json(indent=2))
    _ = tmp.replace(path)


# ---------------------------------------------------------------------------
# Single-run driver
# ---------------------------------------------------------------------------


def _build_providers(
    *,
    pack_runtime: PackedModalRuntime[TestArgsT, CaseSpeedupT],
    provider_config: CompoundProviderConfig,
    observation_type: type[CompoundObservation[CaseSpeedupT]],
) -> tuple[
    CompoundEvaluationProvider[CaseSpeedupT],
    CompoundFeedbackMutationProvider[TestArgsT, CaseSpeedupT],
]:
    pack = pack_runtime.pack
    real_evaluator = GpuModeKernelModalProvider(
        pack_runtime=pack_runtime,
        aggregator=provider_config.aggregator,
        gpu=provider_config.gpu,
        max_in_flight=provider_config.real_evaluator_max_in_flight,
    )
    # AbstainingLlmSpeedupEstimator demands a positive int max_tokens.
    # When the configured surrogate has max_tokens=None (Gemini-3-flash
    # rejects an explicit cap), pass a generous ceiling that the prompt
    # comfortably fits inside. The estimator constructor stores it, but
    # whether it's threaded into the LiteLLM call depends on the
    # estimator's own logic — keep parity with the always-real harness.
    surrogate_max_tokens_effective = (
        provider_config.surrogate.max_tokens
        if provider_config.surrogate.max_tokens is not None
        else 32000
    )
    surrogate = AbstainingLlmSpeedupEstimator(
        model_slug=provider_config.surrogate.model_slug,
        temperature=provider_config.surrogate.temperature,
        max_tokens=surrogate_max_tokens_effective,
        request_timeout_s=provider_config.surrogate.request_timeout_s,
        num_retries=provider_config.surrogate.num_retries,
    )
    evaluation_provider = CompoundEvaluationProvider[CaseSpeedupT](
        surrogate=surrogate,
        real_evaluator=real_evaluator,
        forecast_reward=ExpectedSpeedupReward(),
        task=KernelTaskInfo(op_name=pack.name, level_id=0, task_id=0),
        reference=KernelImplementation(
            kernel_name=f"{pack.name}_pytorch_reference",
            code=pack.seed_kernel_code,
            runtime_ms=None,
        ),
        hardware=provider_config.hardware,
        observation_type=observation_type,
        candidate_kernel_name=f"{pack.name}_candidate",
        max_surrogate_concurrency=provider_config.surrogate.max_concurrency,
    )
    mutation_provider = CompoundFeedbackMutationProvider[TestArgsT, CaseSpeedupT](
        pack=pack,
        model_slug=provider_config.mutator.model_slug,
        gpu_name=provider_config.gpu,
        max_llm_concurrency=provider_config.mutator.max_concurrency,
        request_timeout_s=provider_config.mutator.request_timeout_s,
        max_tokens=provider_config.mutator.max_tokens,
        temperature=provider_config.mutator.temperature,
        num_retries=provider_config.mutator.num_retries,
    )
    return evaluation_provider, mutation_provider


def _run_single(
    *,
    pack_runtime: PackedModalRuntime[TestArgsT, CaseSpeedupT],
    observation_type: type[CompoundObservation[CaseSpeedupT]],
    run_dir: Path,
    run_config: CompoundRunConfig,
) -> None:
    """Execute one v2 compound search into ``run_dir``."""
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "events.jsonl"
    pack_name = pack_runtime.pack.name
    seed_program = pack_runtime.pack.seed_kernel_code

    event_log: FileEventLog[CompoundObservation[CaseSpeedupT]] = FileEventLog(
        log_path, observation_type=observation_type
    )

    evaluation_provider, mutation_provider = _build_providers(
        pack_runtime=pack_runtime,
        provider_config=run_config.provider,
        observation_type=observation_type,
    )

    logger.info(
        "v2 compound {pack} run starting in {dir}: steps={steps} batch={batch} spp={spp} k={k} surrogate={s} mutator={m} gpu={gpu}",
        pack=pack_name,
        dir=run_dir,
        steps=run_config.search.total_budget_steps,
        batch=run_config.search.batch_size,
        spp=run_config.search.samples_per_parent,
        k=run_config.search.k_per_parent,
        s=run_config.provider.surrogate.model_slug,
        m=run_config.provider.mutator.model_slug,
        gpu=run_config.provider.gpu,
    )

    start_s = time.perf_counter()
    with mutation_provider, evaluation_provider:
        driver = SearchDriver[CompoundObservation[CaseSpeedupT]](
            run_config.search,
            mutation_provider=mutation_provider,
            evaluation_provider=evaluation_provider,
            event_log=event_log,
            observation_type=observation_type,
        )
        _ = driver.run(initial_program=seed_program)
    wall_clock_seconds = time.perf_counter() - start_s

    summary = _compute_summary(
        event_log,
        run_config.search,
        wall_clock_seconds,
        observation_type=observation_type,
    )
    _atomic_write_json(summary, run_dir / "summary.json")

    bf = (
        f"{summary.best_forecast_reward:.4f}x"
        if summary.best_forecast_reward is not None
        else "none"
    )
    bc = (
        f"{summary.best_confirmed_reward:.4f}x"
        if summary.best_confirmed_reward is not None
        else "none"
    )
    logger.info(
        "v2 compound {pack} run done in {dir}: steps={steps} archive={size} forecasts={f} real_evals={r} deferrals={d} best_forecast={bf} best_confirmed={bc} wall={wall:.1f}s",
        pack=pack_name,
        dir=run_dir,
        steps=summary.steps_completed,
        size=summary.archive_size,
        f=summary.n_forecasts,
        r=summary.n_real_evals,
        d=summary.n_deferrals,
        bf=bf,
        bc=bc,
        wall=wall_clock_seconds,
    )


# ---------------------------------------------------------------------------
# Multi-run driver — the experiment-level entry point.
# ---------------------------------------------------------------------------


def run_compound_pack_experiment(
    *,
    pack_runtime: PackedModalRuntime[TestArgsT, CaseSpeedupT],
    case_speedup_type: type[CaseSpeedupT],
    output_dir: Path,
    config: CompoundExperimentConfig,
) -> None:
    """Run ``config.num_runs`` independent compound searches into ``output_dir``.

    The observation type
    (``CompoundObservation[case_speedup_type]``) is derived from
    ``case_speedup_type`` by Pydantic generic subscription.
    Already-completed runs (those with a ``summary.json``) are
    skipped, making the entry point safe to re-invoke after a crash.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(config, output_dir / "experiment.json")

    observation_type = cast(
        type[CompoundObservation[CaseSpeedupT]],
        CompoundObservation[case_speedup_type],
    )

    pack_name = pack_runtime.pack.name
    logger.info(
        "v2 compound {pack} experiment starting: surrogate={s} mutator={m} num_runs={n} output_dir={dir}",
        pack=pack_name,
        s=config.run.provider.surrogate.model_slug,
        m=config.run.provider.mutator.model_slug,
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
            run_config=config.run,
        )

    logger.info(
        "v2 compound {pack} experiment done: {dir}", pack=pack_name, dir=output_dir
    )


# ---------------------------------------------------------------------------
# Result-loader helper for per-experiment results.py modules.
# ---------------------------------------------------------------------------


def load_compound_run_summaries(root: Path) -> list[CompoundRunSummary]:
    """Read every ``run_*/summary.json`` under ``root`` into ``CompoundRunSummary``s."""
    if not root.exists():
        return []
    return [
        CompoundRunSummary.model_validate_json(p.read_text())
        for p in sorted(root.glob("run_*/summary.json"))
    ]


__all__ = [
    "CompoundExperimentConfig",
    "CompoundProviderConfig",
    "CompoundRunConfig",
    "CompoundRunSummary",
    "LlmConfig",
    "load_compound_run_summaries",
    "run_compound_pack_experiment",
]
