"""Domain-object → prompt-string rendering.

The right projection of an ``ExperimentResult`` into a prompt is
expected to take iteration; both renderers are exposed via Protocols
so the LLM-backed builder can be swapped to a different rendering
strategy without changing its parser or retry loop.

Defaults try hard to land in a sensible budget without dropping the
information the builder actually needs:

* The world model is rendered verbatim — its size is bounded by the
  builder's own diff edits.
* For the experiment result, every trial is summarised on one line.
  The full code of the *best* trial is included; failing trials get
  a one-line failure summary.
"""

from __future__ import annotations

from typing import Generic, Protocol

from arid_badger.hill_climbing.domain import ObservationT
from arid_badger.hill_climbing.scoring_providers.trimul import TriMulObservation
from arid_badger.theory_builder.v1.domain import (
    ExperimentResult,
    ExperimentTrial,
    WorldModel,
)
from arid_badger.trimul.core import (
    CompileFailedFeedback,
    IncorrectFeedback,
    InfrastructureFailureFeedback,
    RuntimeErrorFeedback,
    SuccessFeedback,
)


class WorldModelRenderer(Protocol):
    def render(self, world_model: WorldModel) -> str: ...


class ExperimentResultRenderer(Protocol[ObservationT]):
    def render(self, result: ExperimentResult[ObservationT]) -> str: ...


class MarkdownWorldModelRenderer:
    """Renders the world model verbatim, prefixed with the kernel
    description.

    The structure (Established Beliefs / Working Hypotheses / Open
    Questions / Anomalies) is a prompt convention; this renderer
    doesn't enforce it."""

    def render(self, world_model: WorldModel) -> str:
        if not world_model.text:
            body = "*(world model is empty — propose your first hypothesis)*"
        else:
            body = world_model.text
        return (
            f"## Kernel under study\n\n{world_model.kernel_description}\n\n"
            f"## Current world model\n\n{body}"
        )


def _summarize_trimul_trial(trial: ExperimentTrial[TriMulObservation]) -> str:
    """One-line summary of a single TriMul trial."""
    feedback = trial.evaluation.observation.feedback
    reward = trial.evaluation.reward
    if reward is not None and isinstance(feedback, SuccessFeedback):
        per_case = feedback.per_case_speedups
        slowest = min(
            (c.speedup for c in per_case), default=float("nan")
        )
        fastest = max(
            (c.speedup for c in per_case), default=float("nan")
        )
        return (
            f"reward={reward:.3f}x (geomean), "
            f"per-case range {slowest:.3f}-{fastest:.3f}, "
            f"{len(per_case)} cases"
        )
    if isinstance(feedback, CompileFailedFeedback):
        return "compile_failed"
    if isinstance(feedback, RuntimeErrorFeedback):
        return f"runtime_error({feedback.runtime_error_name})"
    if isinstance(feedback, IncorrectFeedback):
        return "incorrect"
    if isinstance(feedback, InfrastructureFailureFeedback):
        return f"infra_failure({feedback.reason[:60]})"
    return "unknown"


class TriMulExperimentResultRenderer(Generic[ObservationT]):
    """Renders an ``ExperimentResult[TriMulObservation]`` into a
    prompt string.

    Includes:

    * One-line summary of every trial.
    * Per-case breakdown of the best trial.
    * Full code of the best valid trial.
    * Full code of the best *failing* trial when no trial was valid
      (so the builder still sees what the inner search produced).
    """

    def __init__(self, *, max_code_chars: int = 12000) -> None:
        self._max_code_chars = max_code_chars

    def render(self, result: ExperimentResult[TriMulObservation]) -> str:  # type: ignore[override]
        lines: list[str] = []
        lines.append(
            f"Inner search ran {result.num_trials} trial(s) "
            f"({result.num_valid_trials} valid)."
        )
        lines.append("")
        lines.append("### Trial summaries (in submission order)")
        for i, trial in enumerate(result.trials):
            lines.append(f"  {i}. {_summarize_trimul_trial(trial)}")
        lines.append("")

        best = result.best_trial
        if best is not None and isinstance(
            best.evaluation.observation.feedback, SuccessFeedback
        ):
            best_feedback = best.evaluation.observation.feedback
            lines.append("### Best trial — per-case breakdown")
            sorted_cases = sorted(
                best_feedback.per_case_speedups, key=lambda c: c.speedup
            )
            for case in sorted_cases:
                ref_us = case.ref_runtime_ns / 1_000.0
                cand_us = case.runtime_ns / 1_000.0
                lines.append(
                    f"  seqlen={case.seqlen} bs={case.bs} dim={case.dim} "
                    f"hidden={case.hiddendim} nomask={case.nomask} "
                    f"dist={case.distribution}: "
                    f"{case.speedup:.3f}x "
                    f"(ref {ref_us:.1f}μs, cand {cand_us:.1f}μs)"
                )
            lines.append("")
            lines.append("### Best trial — code")
            code = best.code
            if len(code) > self._max_code_chars:
                code = code[: self._max_code_chars] + "\n# ...[truncated]"
            lines.append(f"```python\n{code}\n```")
        elif result.trials:
            # No valid trial — fall back to the first one so the LLM
            # at least sees one concrete failure.
            ref = result.trials[0]
            lines.append(
                "### No valid trial. First trial code (for diagnosis):"
            )
            code = ref.code
            if len(code) > self._max_code_chars:
                code = code[: self._max_code_chars] + "\n# ...[truncated]"
            lines.append(f"```python\n{code}\n```")
        else:
            lines.append("### Inner search produced no trials.")
        return "\n".join(lines)


__all__ = [
    "WorldModelRenderer",
    "ExperimentResultRenderer",
    "MarkdownWorldModelRenderer",
    "TriMulExperimentResultRenderer",
]
