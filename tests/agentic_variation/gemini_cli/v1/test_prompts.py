"""Unit tests for prompt rendering and feedback summary formatting."""

from __future__ import annotations

from arid_badger.agentic_variation.gemini_cli.v1.prompts import (
    format_feedback_summary,
    render_system_prompt,
    render_user_prompt,
)
from arid_badger.trimul.core import (
    CaseSpeedup,
    CompileFailedFeedback,
    IncorrectFeedback,
    RuntimeErrorFeedback,
    SuccessFeedback,
)


def test_format_feedback_summary_compile_failed() -> None:
    summary = format_feedback_summary(
        CompileFailedFeedback(compilation_error="undefined name 'foo'")
    )
    assert "failed to compile" in summary
    assert "undefined name 'foo'" in summary
    assert "fix the errors" in summary


def test_format_feedback_summary_runtime_error() -> None:
    summary = format_feedback_summary(
        RuntimeErrorFeedback(
            runtime_error_name="RuntimeError",
            runtime_error="something exploded",
            traceback="Traceback (most recent call last):\n  File x",
        )
    )
    assert "exception at runtime" in summary
    assert "RuntimeError" in summary
    assert "something exploded" in summary


def test_format_feedback_summary_incorrect() -> None:
    summary = format_feedback_summary(
        IncorrectFeedback(error_message="max abs diff 0.31 exceeds tol")
    )
    assert "incorrect output" in summary
    assert "max abs diff 0.31 exceeds tol" in summary


def _case(speedup: float, seqlen: int) -> CaseSpeedup:
    return CaseSpeedup(
        seqlen=seqlen,
        bs=1,
        dim=128,
        hiddendim=128,
        nomask=True,
        distribution="normal",
        speedup=speedup,
        runtime_ns=1_000_000.0 / speedup,
        ref_runtime_ns=1_000_000.0,
    )


def test_format_feedback_summary_success_sorts_slowest_first() -> None:
    feedback = SuccessFeedback(
        aggregated_speedup=1.42,
        aggregation_method="geomean",
        per_case_speedups=[
            _case(2.0, 256),
            _case(0.8, 1024),
            _case(1.4, 512),
        ],
    )
    summary = format_feedback_summary(feedback)

    assert "Aggregated speedup: 1.420x" in summary
    assert "geomean" in summary

    # Slowest first: 0.8x (seqlen=1024) should appear before 1.4x and 2.0x.
    idx_slow = summary.index("0.800x")
    idx_mid = summary.index("1.400x")
    idx_fast = summary.index("2.000x")
    assert idx_slow < idx_mid < idx_fast


def test_render_system_prompt_uses_passed_variables() -> None:
    out = render_system_prompt(mcp_tool_name="mcp_foo_bar", benchmark_budget=37)
    assert "mcp_foo_bar" in out
    # The budget is embedded in a markdown-bold span; just check the digits.
    assert "37" in out


def test_render_user_prompt_embeds_seed_and_verdict() -> None:
    seed = SuccessFeedback(
        aggregated_speedup=1.0,
        aggregation_method="geomean",
        per_case_speedups=[_case(1.0, 256)],
    )
    out = render_user_prompt(
        gpu_name="H100",
        triton_version="3.3.1",
        seed_source="def custom_kernel(data):\n    return data[0]\n",
        seed_feedback=seed,
        benchmark_budget=10,
    )
    assert "H100" in out
    assert "3.3.1" in out
    assert "custom_kernel" in out
    assert "Aggregated speedup: 1.000x" in out
