"""Generic experiment harness for pack-based v2 PUCT searches.

Lifted from the duplicated run.py + models.py across e0077
(cross_entropy), e0078 (GDN h), e0079 (GDN o). All three differ only
in (a) the ``PackedModalRuntime`` constant they consume, (b) the
``CaseSpeedupT`` they thread through generic types, and (c) a few
log-message strings. Per the architecture / library-evolution
guidance, three consumers of an identical pattern is the trigger to
promote.

Each pack-based experiment now collapses to ~30 lines: imports plus
a ``CONFIG = ExperimentConfig(...)`` instantiation, with
``run_pack_experiment`` doing the rest.

Resumability is inherited from the v2 driver: if a run is
interrupted, restarting picks up from the event log; runs whose
``summary.json`` already exists are skipped entirely. Variance
across runs comes from LLM sampling at temperature=1.0 — there is
no PRNG seed to thread.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, cast

from loguru import logger
from pydantic import BaseModel

from gpu_forecasters.gpu_mode_kernel.aggregation import AggregationMethod
from gpu_forecasters.gpu_mode_kernel.core import (
    CaseSpeedupT,
    GpuModeKernelObservation,
)
from gpu_forecasters.gpu_mode_kernel.kernel_pack import TestArgsT
from gpu_forecasters.gpu_mode_kernel.modal_scoring import PackedModalRuntime
from gpu_forecasters.gpu_mode_kernel.providers.v2_feedback_mutation import (
    GpuModeKernelFeedbackMutationProvider,
)
from gpu_forecasters.gpu_mode_kernel.providers.v2_modal_scoring import (
    GpuModeKernelModalProvider,
)
from gpu_forecasters.invocation_sink import FilesystemInvocationSink
from gpu_forecasters.max_reward_puct.v2.config import SearchConfig as V2SearchConfig
from gpu_forecasters.max_reward_puct.v2.event_log import FileEventLog
from gpu_forecasters.max_reward_puct.v2.events import (
    EvaluationCompleted,
    EvaluationFailed,
    MutationFailed,
    SearchInitialized,
)
from gpu_forecasters.max_reward_puct.v2.search import SearchDriver
from gpu_forecasters.max_reward_puct.v2.state import replay


# ---------------------------------------------------------------------------
# Config schemas
# ---------------------------------------------------------------------------


class ProviderConfig(BaseModel, frozen=True):
    """Knobs that bind to the concrete v2 providers (litellm + Modal).

    ``max_tokens=None`` is the right setting for Gemini 3 Flash — it
    rejects an explicit cap. Together-hosted gpt-oss requires an
    explicit cap (use 32000 per the e0058/e0060 lineage).
    ``request_timeout_s`` bounds each litellm acompletion before
    litellm retries.
    """

    model_slug: str
    gpu: str
    aggregator: AggregationMethod
    max_llm_concurrency: int
    max_tokens: int | None
    request_timeout_s: float


class RunConfig(BaseModel, frozen=True):
    """One PUCT search worth of config: search shape + provider knobs.

    ``track_invocations`` enables per-evaluation cost/result records via a
    ``FilesystemInvocationSink`` written to ``run_dir/invocations/*.json``.
    Off by default to preserve byte-for-byte parity with prior runs.
    """

    search: V2SearchConfig
    provider: ProviderConfig
    track_invocations: bool = False


class ExperimentConfig(BaseModel, frozen=True):
    """Top-level experiment config: how many runs, and one shared RunConfig."""

    num_runs: int
    run: RunConfig


class RunSummary(BaseModel, frozen=True):
    """End-of-run snapshot derived from the event log.

    ``best_reward`` is None when no archived node has a reward (no
    correct kernel produced). ``seed_reward`` lets downstream readers
    compute speedup-over-seed without re-replaying the log.
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
    wall_clock_seconds: float


# ---------------------------------------------------------------------------
# Per-run summary computation (pure function of an event log on disk)
# ---------------------------------------------------------------------------


def _compute_summary(
    event_log: FileEventLog[GpuModeKernelObservation[CaseSpeedupT]],
    search_config: V2SearchConfig,
    wall_clock_seconds: float,
    observation_type: type[GpuModeKernelObservation[CaseSpeedupT]],
) -> RunSummary:
    events = event_log.read_all()
    state = replay(
        events,
        k_per_parent=search_config.k_per_parent,
        archive_capacity=search_config.archive_capacity,
        observation_type=observation_type,
    )

    num_eval_completed = 0
    num_eval_correct = 0
    num_eval_failed = 0
    num_mut_failed = 0
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

    best = state.best_archived_node()
    return RunSummary(
        steps_completed=state.current_step,
        archive_size=len(state.archive),
        best_reward=None if best is None else best.evaluation.reward,
        seed_reward=seed_reward,
        best_node_ulid=None if best is None else str(best.ulid),
        num_evaluations_total=num_eval_completed,
        num_evaluations_correct=num_eval_correct,
        num_evaluation_failures=num_eval_failed,
        num_mutation_failures=num_mut_failed,
        wall_clock_seconds=wall_clock_seconds,
    )


def _atomic_write_json(model: BaseModel, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    _ = tmp.write_text(model.model_dump_json(indent=2))
    _ = tmp.replace(path)


# ---------------------------------------------------------------------------
# Single-run driver
# ---------------------------------------------------------------------------


def _run_single(
    *,
    pack_runtime: PackedModalRuntime[TestArgsT, CaseSpeedupT],
    observation_type: type[GpuModeKernelObservation[CaseSpeedupT]],
    run_dir: Path,
    run_config: RunConfig,
) -> None:
    """Execute one v2 PUCT search into ``run_dir``."""
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "events.jsonl"

    event_log: FileEventLog[GpuModeKernelObservation[CaseSpeedupT]] = FileEventLog(
        log_path, observation_type=observation_type
    )

    provider = run_config.provider
    search = run_config.search
    pack_name = pack_runtime.pack.name

    mutation_provider = GpuModeKernelFeedbackMutationProvider(
        pack=pack_runtime.pack,
        model_slug=provider.model_slug,
        gpu_name=provider.gpu,
        max_llm_concurrency=provider.max_llm_concurrency,
        request_timeout_s=provider.request_timeout_s,
        max_tokens=provider.max_tokens,
    )
    invocation_sink = (
        FilesystemInvocationSink(run_dir / "invocations")
        if run_config.track_invocations
        else None
    )
    evaluation_provider = GpuModeKernelModalProvider(
        pack_runtime=pack_runtime,
        aggregator=provider.aggregator,
        gpu=provider.gpu,
        max_in_flight=provider.max_llm_concurrency,
        invocation_sink=invocation_sink,
    )

    logger.info(
        "v2 {pack} run starting in {dir}: steps={steps} batch={batch} "
        "spp={spp} k={k} model={model} max_tok={mt} timeout={to:.0f}s",
        pack=pack_name,
        dir=run_dir,
        steps=search.total_budget_steps,
        batch=search.batch_size,
        spp=search.samples_per_parent,
        k=search.k_per_parent,
        model=provider.model_slug,
        mt=provider.max_tokens,
        to=provider.request_timeout_s,
    )

    start_s = time.perf_counter()
    with mutation_provider, evaluation_provider:
        driver = SearchDriver[GpuModeKernelObservation[CaseSpeedupT]](
            search,
            mutation_provider=mutation_provider,
            evaluation_provider=evaluation_provider,
            event_log=event_log,
            observation_type=observation_type,
        )
        _ = driver.run(initial_program=pack_runtime.pack.seed_kernel_code)
    wall_clock_seconds = time.perf_counter() - start_s

    summary = _compute_summary(
        event_log,
        search,
        wall_clock_seconds,
        observation_type=observation_type,
    )
    _atomic_write_json(summary, run_dir / "summary.json")

    best_str = (
        f"{summary.best_reward:.4f}x" if summary.best_reward is not None else "none"
    )
    logger.info(
        "v2 {pack} run done in {dir}: steps={steps} archive={size} "
        "best={best} correct={c}/{n} wall={wall:.1f}s",
        pack=pack_name,
        dir=run_dir,
        steps=summary.steps_completed,
        size=summary.archive_size,
        best=best_str,
        c=summary.num_evaluations_correct,
        n=summary.num_evaluations_total,
        wall=wall_clock_seconds,
    )


# ---------------------------------------------------------------------------
# Multi-run driver — the experiment-level entry point.
# ---------------------------------------------------------------------------


def run_pack_experiment(
    *,
    pack_runtime: PackedModalRuntime[TestArgsT, CaseSpeedupT],
    case_speedup_type: type[CaseSpeedupT],
    output_dir: Path,
    config: ExperimentConfig,
) -> None:
    """Run ``config.num_runs`` independent v2 PUCT searches into ``output_dir``.

    The observation type (``GpuModeKernelObservation[case_speedup_type]``)
    is derived from ``case_speedup_type`` by Pydantic generic
    subscription. Pydantic v2 resolves the subscripted alias at
    runtime — the same mechanism the per-pack experiment files use
    when they define their own observation alias.

    Already-completed runs (those with a ``summary.json``) are
    skipped, making the entry point safe to re-invoke after a crash.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(config, output_dir / "experiment.json")

    observation_type = cast(
        type[GpuModeKernelObservation[CaseSpeedupT]],
        GpuModeKernelObservation[case_speedup_type],
    )

    pack_name = pack_runtime.pack.name
    logger.info(
        "v2 {pack} experiment starting: model={model} num_runs={n} "
        "output_dir={dir}",
        pack=pack_name,
        model=config.run.provider.model_slug,
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

    logger.info("v2 {pack} experiment done: {dir}", pack=pack_name, dir=output_dir)


# ---------------------------------------------------------------------------
# Result-loader helper for per-experiment results.py modules.
# ---------------------------------------------------------------------------


def load_run_summaries(root: Path) -> list[RunSummary]:
    """Read every ``run_*/summary.json`` under ``root`` into ``RunSummary``s.

    Used by per-experiment ``results.py`` loaders. Returns ``[]`` if
    the directory does not yet exist (caller hasn't produced any
    output yet).
    """
    if not root.exists():
        return []
    return [
        RunSummary.model_validate_json(p.read_text())
        for p in sorted(root.glob("run_*/summary.json"))
    ]


# Re-export the bound types for the (rare) caller that wants them:
# the public surface is intentionally small but ``Any`` would be a
# regression vs. the per-experiment classes' typed loaders.
__all__ = [
    "ExperimentConfig",
    "ProviderConfig",
    "RunConfig",
    "RunSummary",
    "load_run_summaries",
    "run_pack_experiment",
]


# Suppress unused import warning — ``Any`` is only present in the
# type annotations Pydantic introspects, not in the function bodies.
_ = Any
