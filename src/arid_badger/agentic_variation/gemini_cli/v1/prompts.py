"""Prompt rendering and feedback formatting for the Gemini CLI agent.

Two Jinja templates in this directory — ``system_prompt.md.j2`` (role /
methodology / tool guide, generic over any multi-turn Triton-kernel
optimization task) and ``user_prompt.md.j2`` (the TriMul task instance
handed to the agent on turn 1) — are the only prompt surfaces. No
``GEMINI.md``, no inline Python strings scattered across the orchestrator.

Everything here is deliberately duplicated from the library's non-agentic
path (``arid_badger.hill_climbing.mutation_providers.trimul_feedback_mutation``)
rather than imported. A prompt tweak that helps the agent could easily
regress the evolutionary search, so the two paths own their prompts
independently. The vendored TriMul task body lives in
``user_prompt.md.j2``; changes to the canonical task description should
flow into both copies.
"""

from __future__ import annotations

from pathlib import Path

from arid_badger.trimul.core import (
    CompileFailedFeedback,
    IncorrectFeedback,
    RuntimeErrorFeedback,
    TriMulKernelExecutionFeedback,
)
from jinja2 import Environment, FileSystemLoader, StrictUndefined


# Truncation budgets for error / traceback text that gets embedded in
# feedback summaries. Copied from the library's formatter; owned here so
# we can tune them for the agent's multi-turn context independently of
# the one-shot mutation operator.
_MAX_COMPILATION_ERROR_CHARS = 2000
_MAX_RUNTIME_ERROR_CHARS = 1000
_MAX_TRACEBACK_CHARS = 3000
_MAX_INCORRECT_ERROR_CHARS = 2000


_TEMPLATES_DIR = Path(__file__).parent
_env = Environment(
    loader=FileSystemLoader(_TEMPLATES_DIR),
    # StrictUndefined turns "forgot to pass a variable" into a loud
    # render-time error instead of silently emitting an empty string.
    undefined=StrictUndefined,
    keep_trailing_newline=True,
)


def _truncate_head(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n[truncated]"


def _truncate_tail(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return f"[truncated]\n...{text[-max_chars:]}"


def format_feedback_summary(feedback: TriMulKernelExecutionFeedback) -> str:
    """Render one TriMul evaluation verdict as natural-language text.

    Used in two places:

    - ``render_user_prompt`` bakes the seed kernel's verdict into turn 1
      so the agent doesn't burn a tool call on the seed.
    - ``trimul_score_server.score_trimul`` returns this as the
      ``summary`` field of every subsequent benchmark response.

    The shape mirrors the feedback rendering the non-agentic TriMul
    mutation operator puts in front of its one-shot model (per-case
    breakdown sorted slowest-first, closing "rewrite the entire kernel"
    / "fix the errors" directive), so the agent's signal surface matches
    the mutation operator's.
    """
    if isinstance(feedback, CompileFailedFeedback):
        return (
            "Your kernel failed to compile.\n\n"
            "Compilation error:\n"
            f"{_truncate_head(feedback.compilation_error, _MAX_COMPILATION_ERROR_CHARS)}\n\n"
            "Please fix the errors and try again."
        )
    if isinstance(feedback, RuntimeErrorFeedback):
        return (
            "Your kernel raised an exception at runtime.\n\n"
            f"Error type: {feedback.runtime_error_name}\n\n"
            "Error message:\n"
            f"{_truncate_head(feedback.runtime_error, _MAX_RUNTIME_ERROR_CHARS)}\n\n"
            "Traceback:\n"
            f"{_truncate_tail(feedback.traceback, _MAX_TRACEBACK_CHARS)}\n\n"
            "Please fix the errors and try again."
        )
    if isinstance(feedback, IncorrectFeedback):
        return (
            "Your kernel produced incorrect output compared to the reference.\n\n"
            "Correctness issue:\n"
            f"{_truncate_head(feedback.error_message, _MAX_INCORRECT_ERROR_CHARS)}\n\n"
            "Please fix the correctness issues and try again."
        )
    # SuccessFeedback — the only remaining variant of the 4-kind union.
    sorted_cases = sorted(feedback.per_case_speedups, key=lambda c: c.speedup)
    lines: list[str] = [
        "You are iteratively optimizing runtime (microseconds).",
        "",
        (
            f"Your kernel is correct. "
            f"Aggregated speedup: {feedback.aggregated_speedup:.3f}x "
            f"(aggregation method: {feedback.aggregation_method})."
        ),
        "",
        "Per-case breakdown (slowest first):",
    ]
    for case in sorted_cases:
        ref_us = case.ref_runtime_ns / 1_000.0
        candidate_us = case.runtime_ns / 1_000.0
        lines.append(
            f"  seqlen={case.seqlen}, bs={case.bs}, dim={case.dim}, "
            f"hiddendim={case.hiddendim}, nomask={case.nomask}, "
            f"dist={case.distribution}: "
            f"{case.speedup:.3f}x "
            f"(ref: {ref_us:.1f}\u03bcs, candidate: {candidate_us:.1f}\u03bcs)"
        )
    lines.append("")
    lines.append(
        "Please rewrite the entire kernel to be as fast as possible. "
        "Focus on the slowest configurations listed above."
    )
    return "\n".join(lines)


def render_system_prompt(*, mcp_tool_name: str, benchmark_budget: int) -> str:
    """Render the agent's system prompt (role, methodology, tool guide)."""
    return _env.get_template("system_prompt.md.j2").render(
        mcp_tool_name=mcp_tool_name,
        benchmark_budget=benchmark_budget,
    )


def render_user_prompt(
    *,
    gpu_name: str,
    triton_version: str,
    seed_source: str,
    seed_feedback: TriMulKernelExecutionFeedback,
    benchmark_budget: int,
) -> str:
    """Render turn 1's user message: task + rules + seed source + verdict."""
    return _env.get_template("user_prompt.md.j2").render(
        gpu_name=gpu_name,
        triton_version=triton_version,
        seed_source=seed_source,
        seed_verdict_summary=format_feedback_summary(seed_feedback),
        benchmark_budget=benchmark_budget,
    )
