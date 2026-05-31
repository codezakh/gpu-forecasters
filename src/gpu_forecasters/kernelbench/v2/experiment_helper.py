"""Generic experiment harness for v2 KernelBench L3 PUCT searches.

Promoted from ``experiments/e0123_kernelbench_v2_mutation_smoke/search_smoke.py``
per the library-evolution skill: the upcoming Tier-B per-problem cell
batch is the second consumer of the same wiring (load reference code,
build event log, build mutation + eval providers, open both as context
managers, drive ``SearchDriver[KernelBenchObservation]``, replay log,
write ``summary.json``), so the pattern lifts here.

Mirrors ``arid_badger.gpu_mode_kernel.experiment_helper.run_pack_experiment``
in shape (``ExperimentConfig`` / ``RunConfig`` / ``ProviderConfig`` /
``RunSummary`` / skip-if-summary-present). Differs on three points:

- Takes an ``L3ProblemReference`` (not a ``KernelPack``); the helper
  stays decoupled from the Tier-B registry so callers can pass an
  off-registry problem (prototype, spot-check) without touching
  ``TIER_B_PROBLEMS``.
- Splits ``max_llm_concurrency`` (mutation) from ``max_eval_in_flight``
  (Modal scoring). The two providers do not share a concurrency budget;
  conflating them under-utilizes one or over-pressures the other.
- ``ProviderConfig.gpu`` is typed as ``GpuKind`` so cells get
  type-checked typo detection at the cell-config call site.

When ``track_invocations=True`` a ``FilesystemInvocationSink`` is wired
into **both** the mutation provider and the eval provider so the run
directory's ``invocations/*.json`` carries both LLM-call cost telemetry
and per-evaluation Modal wall-clock.
"""

from __future__ import annotations

import time
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from arid_badger.hill_climbing.scoring_providers.kernelbench import (
    KernelBenchObservation,
)
from arid_badger.invocation_sink import FilesystemInvocationSink
from arid_badger.kernelbench.v2.l3_problems import L3ProblemReference
from arid_badger.kernelbench.v2.providers.kernel_execution_feedback import (
    KernelBenchFeedbackMutationProvider,
)
from arid_badger.kernelbench.v2.providers.modal_scoring import (
    KernelBenchModalProvider,
)
from arid_badger.max_reward_puct.v2.config import SearchConfig as V2SearchConfig
from arid_badger.max_reward_puct.v2.event_log import FileEventLog
from arid_badger.max_reward_puct.v2.events import (
    EvaluationCompleted,
    EvaluationFailed,
    MutationFailed,
    SearchInitialized,
)
from arid_badger.max_reward_puct.v2.search import SearchDriver
from arid_badger.max_reward_puct.v2.state import replay
from arid_badger.modal_gpu import GpuKind


# ---------------------------------------------------------------------------
# Config schemas
# ---------------------------------------------------------------------------


class ProviderConfig(BaseModel, frozen=True):
    """Knobs that bind to the v2 KernelBench providers.

    ``max_tokens=None`` is the right setting for Gemini 3 Flash — it
    rejects an explicit cap. Together-hosted gpt-oss requires an
    explicit cap (use 32000 per the e0058/e0060/e0082 lineage).
    ``max_llm_concurrency`` bounds simultaneous outbound LLM calls;
    ``max_eval_in_flight`` bounds simultaneous in-flight Modal eval
    pipelines (compile+bench). The two are decoupled because the
    mutation provider parks on ``litellm.acompletion`` await points
    while the eval provider parks on Modal ``.remote.aio`` await
    points — they do not contend.
    """

    model_slug: str
    gpu: GpuKind
    backend: str = "cuda"
    precision: str = "fp32"
    max_llm_concurrency: int
    max_eval_in_flight: int
    max_tokens: int | None
    request_timeout_s: float
    num_correct_trials: int = 5
    num_perf_trials: int = 100


class RunConfig(BaseModel, frozen=True):
    """One PUCT search worth of config.

    ``track_invocations`` enables per-mutation and per-evaluation
    telemetry via a ``FilesystemInvocationSink`` written to
    ``run_dir/invocations/*.json``. Off by default to preserve
    byte-for-byte parity with smokes that did not record telemetry.
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
    event_log: FileEventLog[KernelBenchObservation],
    search_config: V2SearchConfig,
    wall_clock_seconds: float,
) -> RunSummary:
    events = event_log.read_all()
    state = replay(
        events,
        k_per_parent=search_config.k_per_parent,
        archive_capacity=search_config.archive_capacity,
        observation_type=KernelBenchObservation,
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
    problem: L3ProblemReference,
    run_dir: Path,
    run_config: RunConfig,
) -> None:
    """Execute one v2 PUCT search for ``problem`` into ``run_dir``."""
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = run_dir / "events.jsonl"

    event_log: FileEventLog[KernelBenchObservation] = FileEventLog(
        log_path, observation_type=KernelBenchObservation
    )

    provider = run_config.provider
    search = run_config.search

    invocation_sink = (
        FilesystemInvocationSink(run_dir / "invocations")
        if run_config.track_invocations
        else None
    )

    mutation_provider = KernelBenchFeedbackMutationProvider(
        reference_kernel_code=problem.reference_kernel_code,
        model_slug=provider.model_slug,
        max_llm_concurrency=provider.max_llm_concurrency,
        request_timeout_s=provider.request_timeout_s,
        max_tokens=provider.max_tokens,
        invocation_sink=invocation_sink,
    )
    evaluation_provider = KernelBenchModalProvider(
        reference_kernel_code=problem.reference_kernel_code,
        gpu=provider.gpu,
        backend=provider.backend,
        precision=provider.precision,
        num_correct_trials=provider.num_correct_trials,
        num_perf_trials=provider.num_perf_trials,
        max_in_flight=provider.max_eval_in_flight,
        invocation_sink=invocation_sink,
    )

    logger.info(
        "v2 KernelBench L3 run starting in {dir}: problem={problem} "
        "steps={steps} batch={batch} spp={spp} k={k} model={model} "
        "max_tok={mt} gpu={gpu}",
        dir=run_dir,
        problem=problem.name,
        steps=search.total_budget_steps,
        batch=search.batch_size,
        spp=search.samples_per_parent,
        k=search.k_per_parent,
        model=provider.model_slug,
        mt=provider.max_tokens,
        gpu=provider.gpu.value,
    )

    start_s = time.perf_counter()
    with mutation_provider, evaluation_provider:
        driver = SearchDriver[KernelBenchObservation](
            search,
            mutation_provider=mutation_provider,
            evaluation_provider=evaluation_provider,
            event_log=event_log,
            observation_type=KernelBenchObservation,
        )
        _ = driver.run(initial_program=problem.reference_kernel_code)
    wall_clock_seconds = time.perf_counter() - start_s

    summary = _compute_summary(event_log, search, wall_clock_seconds)
    _atomic_write_json(summary, run_dir / "summary.json")

    best_str = (
        f"{summary.best_reward:.4f}x" if summary.best_reward is not None else "none"
    )
    logger.info(
        "v2 KernelBench L3 run done in {dir}: problem={problem} "
        "steps={steps} archive={size} best={best} correct={c}/{n} "
        "wall={wall:.1f}s",
        dir=run_dir,
        problem=problem.name,
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


def run_l3_experiment(
    *,
    problem: L3ProblemReference,
    output_dir: Path,
    config: ExperimentConfig,
) -> None:
    """Run ``config.num_runs`` independent v2 PUCT searches on ``problem``
    into ``output_dir``.

    Already-completed runs (those with a ``summary.json``) are skipped,
    making the entry point safe to re-invoke after a crash. Variance
    across runs comes from temperature=1.0 LLM sampling — there is no
    PRNG seed to thread.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_json(config, output_dir / "experiment.json")

    logger.info(
        "v2 KernelBench L3 experiment starting: problem={problem} "
        "model={model} num_runs={n} output_dir={dir}",
        problem=problem.name,
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
            problem=problem,
            run_dir=run_dir,
            run_config=config.run,
        )

    logger.info(
        "v2 KernelBench L3 experiment done: problem={problem} {dir}",
        problem=problem.name,
        dir=output_dir,
    )


# ---------------------------------------------------------------------------
# Result-loader helper for per-experiment results.py modules.
# ---------------------------------------------------------------------------


def load_run_summaries(root: Path) -> list[RunSummary]:
    """Read every ``run_*/summary.json`` under ``root`` into ``RunSummary``s.

    Used by per-experiment ``results.py`` loaders. Returns ``[]`` if
    the directory does not yet exist (caller hasn't produced any output
    yet).
    """
    if not root.exists():
        return []
    return [
        RunSummary.model_validate_json(p.read_text())
        for p in sorted(root.glob("run_*/summary.json"))
    ]


__all__ = [
    "ExperimentConfig",
    "ProviderConfig",
    "RunConfig",
    "RunSummary",
    "load_run_summaries",
    "run_l3_experiment",
]
