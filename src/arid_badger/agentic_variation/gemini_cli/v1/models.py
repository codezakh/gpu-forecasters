from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from arid_badger.trimul.core import TriMulKernelExecutionFeedback


ThinkingLevel = Literal["LOW", "MEDIUM", "HIGH"]


class ExperimentConfig(BaseModel):
    """All tunable knobs for one run of the Gemini CLI agentic variation operator.

    Treated as the single source of truth at runtime and as the persisted
    provenance field on ``TrimulRunResult`` — the same object that drove
    the run is what lands in ``result.json``, so the record cannot drift
    from the configuration it claims to describe.
    """

    model_config = ConfigDict(frozen=True, protected_namespaces=())

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

    config: ExperimentConfig
    exit_code: int
    elapsed_s: float
    final_kernel_source: str | None
    best_speedup: float | None
    best_kernel_sha256: str | None
