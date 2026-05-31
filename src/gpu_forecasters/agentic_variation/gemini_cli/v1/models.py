from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal, Protocol

from pydantic import BaseModel, ConfigDict

from arid_badger.trimul.core import TriMulKernelExecutionFeedback

from .prompts import render_system_prompt, render_user_prompt


ThinkingLevel = Literal["LOW", "MEDIUM", "HIGH"]


# ---------------------------------------------------------------------------
# Interface types for the config's pluggable behaviour surface.
#
# Kept alongside :class:`ExperimentConfig` so the config can reference
# them without importing the concrete defaults (which live in
# :mod:`prompts` and :mod:`hooks`). Concrete defaults are bound via
# ``default_factory`` lambdas with local imports to avoid an import cycle.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromptContext:
    """Fields available to a :class:`PromptRenderer`.

    Superset of what either the system or user prompt renderer needs;
    each renderer may consult any subset. ``seed_feedback`` is populated
    from the baseline priming step, so callers building a context
    outside of ``run_experiment`` must have a feedback object on hand.
    """

    mcp_tool_name: str
    benchmark_budget: int
    gpu_name: str
    triton_version: str
    seed_source: str
    seed_feedback: TriMulKernelExecutionFeedback


class PromptRenderer(Protocol):
    def __call__(self, ctx: PromptContext, cfg: ExperimentConfig) -> str: ...


@dataclass(frozen=True)
class PostRunContext:
    """Everything a post-run hook may need to read from or write to.

    ``scratch`` is the container's working directory on the host — where
    the agent wrote ``kernel.py`` and any other artifacts it was asked
    to produce. ``run_artifacts_dir`` is the persistent destination
    (the caller-supplied ``run_dir``). ``config`` is passed so hooks
    that consult experiment knobs don't need to close over it.
    """

    scratch: Path
    run_artifacts_dir: Path
    config: ExperimentConfig


PostRunHook = Callable[[PostRunContext], None]


@dataclass(frozen=True)
class ToolCallContext:
    """Everything a per-tool-call hook needs to react to a completed tool call.

    Fires once per correlated ``tool_use`` + ``tool_result`` pair from
    the Gemini CLI container's NDJSON stream. The library owns the
    correlation machinery (matching on ``tool_id``), so hooks never
    touch the raw event ordering — they see one logical "tool call
    finished" event with both sides in hand, and the on-disk state in
    ``scratch`` already reflects whatever the tool did.

    Match on ``tool_name`` and/or ``parameters`` to pick the calls you
    care about; check ``status`` before reading files the tool was
    supposed to write, since a failed write leaves the on-disk state
    unreliable. ``use_event`` / ``result_event`` are kept alongside the
    extracted fields for hooks that need schema-specific extras the
    library hasn't promoted.
    """

    scratch: Path
    run_artifacts_dir: Path
    config: ExperimentConfig
    tool_name: str
    parameters: dict[str, Any]
    tool_id: str
    status: str
    started_at: str
    finished_at: str
    use_event: dict[str, Any]
    result_event: dict[str, Any]


PerToolCallHook = Callable[[ToolCallContext], None]


def default_system_prompt_renderer(
    ctx: PromptContext, cfg: ExperimentConfig
) -> str:
    """Adapter: render the package's built-in system prompt template.

    Ignores ``cfg`` and the data-only fields of ``ctx`` beyond
    ``mcp_tool_name`` / ``benchmark_budget``.
    """
    del cfg
    return render_system_prompt(
        mcp_tool_name=ctx.mcp_tool_name,
        benchmark_budget=ctx.benchmark_budget,
    )


def default_user_prompt_renderer(
    ctx: PromptContext, cfg: ExperimentConfig
) -> str:
    """Adapter: render the package's built-in user prompt template."""
    del cfg
    return render_user_prompt(
        gpu_name=ctx.gpu_name,
        triton_version=ctx.triton_version,
        seed_source=ctx.seed_source,
        seed_feedback=ctx.seed_feedback,
        benchmark_budget=ctx.benchmark_budget,
    )


def copy_kernel_files(ctx: PostRunContext) -> None:
    """Copy every ``kernel*.py`` from the scratch dir into artifacts.

    The versioned-file discipline (``kernel.py``, ``kernel_v1.py``, ...)
    lets a human browse candidates without parsing JSONL. The server-side
    trajectory log is still the source of truth; this is a convenience.
    """
    for src in sorted(ctx.scratch.glob("kernel*.py")):
        _ = shutil.copy2(src, ctx.run_artifacts_dir / src.name)


@dataclass(frozen=True)
class ExperimentConfig:
    """All tunable knobs for one run of the Gemini CLI agentic variation operator.

    Holds the experiment constants (model, gpu, budgets, …) alongside
    the pluggable behaviour hooks — prompt renderers and post-run hooks
    — so a caller can swap in a different prompt or add a hook that
    extracts extra artifacts (``memory.md``, ``learnings.md``, …) from
    the agent's scratch dir without touching the orchestrator.

    Not round-tripped to disk: each experiment declares its own
    ``CONFIG`` constant in ``__main__.py``, which is the authoritative
    provenance record. ``result.json`` carries the *runtime* outcome only.
    """

    model_slug: str
    gpu: str
    triton_version: str
    max_session_turns: int
    aggregator: str
    # Gemini ``thinkingConfig.thinkingLevel`` enum (``LOW`` / ``MEDIUM`` /
    # ``HIGH``). ``None`` leaves the CLI's built-in alias defaults
    # untouched (``HIGH`` for ``gemini-3-pro-preview``). Gemini 3 Flash
    # does not support thinking (``thinking: false`` in the CLI's model
    # registry), so this field is only meaningful for Pro-family models.
    thinking_level: ThinkingLevel | None = None
    system_prompt_renderer: PromptRenderer = default_system_prompt_renderer
    user_prompt_renderer: PromptRenderer = default_user_prompt_renderer
    post_run_hooks: tuple[PostRunHook, ...] = field(
        default_factory=lambda: (copy_kernel_files,)
    )
    # Fires synchronously inside the container-log consumption loop once
    # per correlated ``tool_use`` + ``tool_result`` pair — i.e. after a
    # tool call has finished executing, with the on-disk scratch state
    # reflecting whatever the tool did. Empty by default; an experiment
    # opts in by passing a tuple of :data:`PerToolCallHook`s. A slow
    # hook blocks log consumption — fine for quick file I/O, a problem
    # for anything doing network / GPU work.
    per_tool_call_hooks: tuple[PerToolCallHook, ...] = ()


class TrajectoryRecord(BaseModel):
    """One ``score_trimul`` call logged from the server's side.

    Written by the scoring server into ``trajectory.jsonl`` on every
    benchmark call. The kernel source is snapshotted at call time so
    post-run analysis does not depend on the agent preserving files on
    disk; the versioned ``kernel_v*.py`` naming is a nicety for the agent,
    this log is the source of truth.
    """

    model_config = ConfigDict(frozen=True)

    timestamp_utc: str
    path: str
    sha256: str
    kernel_source: str
    feedback: TriMulKernelExecutionFeedback


class RepeatedRunSummary(BaseModel):
    """Aggregate across N repetitions of the Gemini CLI harness.

    Written by callers that perform a repeated-run sweep as ``summary.json``
    alongside per-run ``run_<NN>/`` dirs, and also recomputed lazily on
    the read side so a partially-completed sweep has an unambiguous view
    of its own state. ``expected_num_runs`` is the target declared by the
    caller; ``completed_num_runs`` is how many ``run_<NN>/result.json``
    files are currently on disk. ``best_speedup_per_run`` preserves order
    over the completed runs (``None`` entries flag runs that produced no
    successful candidate).
    """

    model_config = ConfigDict(frozen=True)

    expected_num_runs: int
    completed_num_runs: int
    num_with_success: int
    best_speedup_per_run: list[float | None]
    min: float | None
    median: float | None
    max: float | None


class BaselineFeedbackEntry(BaseModel):
    """Cached baseline-kernel verdict for one ``(seed, gpu, triton, aggregator)`` tuple.

    The seed kernel is fixed (``SEED_KERNEL_CODE``) and the baseline
    verdict depends only on the scoring configuration, so every run in a
    repeated-run sweep would otherwise re-run the same Modal benchmark.
    This entry wraps the feedback in a ``BaseModel`` purely so
    :class:`FileCache` (which requires ``BaseModel`` values) can persist
    it per-workspace.
    """

    model_config = ConfigDict(frozen=True)

    feedback: TriMulKernelExecutionFeedback


class TrimulRunResult(BaseModel):
    """One end-to-end run of the Gemini CLI agent against the TriMul task.

    ``final_kernel_source`` is the candidate left in ``kernel.py`` at the
    end of the session (``None`` if the agent never produced one). The
    best-across-all-benchmark-calls verdict is computed from
    ``trajectory.jsonl`` (server-side snapshots); ``best_speedup`` and
    ``best_kernel_sha256`` are ``None`` if no successful candidate was
    benchmarked. The agent's per-turn NDJSON and the scoring server's
    stdout live next to this record as ``agent_raw.log`` / ``server.log``.
    """

    model_config = ConfigDict(frozen=True)

    exit_code: int
    elapsed_s: float
    final_kernel_source: str | None
    best_speedup: float | None
    best_kernel_sha256: str | None
