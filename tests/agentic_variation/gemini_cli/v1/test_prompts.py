"""Unit tests for prompt rendering and feedback summary formatting."""

from __future__ import annotations

from gpu_forecasters.agentic_variation.gemini_cli.v1 import (
    ExperimentConfig,
    PromptContext,
    default_system_prompt_renderer,
    default_user_prompt_renderer,
)
from gpu_forecasters.agentic_variation.gemini_cli.v1.prompts import (
    format_feedback_summary,
    render_system_prompt,
    render_user_prompt,
)
from gpu_forecasters.trimul.core import (
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


# ---------------------------------------------------------------------------
# Renderer Protocol wiring — proves that the ``ExperimentConfig``-level
# indirection reaches the same text as the template-backed functions, and
# that a custom renderer is honoured.
# ---------------------------------------------------------------------------


def _sample_context() -> PromptContext:
    return PromptContext(
        mcp_tool_name="mcp_trimul_score_trimul",
        benchmark_budget=7,
        gpu_name="H100",
        triton_version="3.3.1",
        seed_source="def kernel(): ...\n",
        seed_feedback=SuccessFeedback(
            aggregated_speedup=1.0,
            aggregation_method="geomean",
            per_case_speedups=[_case(1.0, 256)],
        ),
    )


def _sample_config() -> ExperimentConfig:
    return ExperimentConfig(
        model_slug="gemini-3-flash-preview",
        gpu="H100",
        triton_version="3.3.1",
        max_session_turns=7,
        aggregator="geomean",
    )


def test_default_system_renderer_matches_template_function() -> None:
    ctx = _sample_context()
    cfg = _sample_config()
    assert default_system_prompt_renderer(ctx, cfg) == render_system_prompt(
        mcp_tool_name=ctx.mcp_tool_name,
        benchmark_budget=ctx.benchmark_budget,
    )


def test_default_user_renderer_matches_template_function() -> None:
    ctx = _sample_context()
    cfg = _sample_config()
    assert default_user_prompt_renderer(ctx, cfg) == render_user_prompt(
        gpu_name=ctx.gpu_name,
        triton_version=ctx.triton_version,
        seed_source=ctx.seed_source,
        seed_feedback=ctx.seed_feedback,
        benchmark_budget=ctx.benchmark_budget,
    )


def test_custom_renderer_is_respected_via_config() -> None:
    sentinel = "CUSTOM-PROMPT-TEXT"

    def custom(ctx: PromptContext, cfg: ExperimentConfig) -> str:
        del ctx, cfg
        return sentinel

    cfg = ExperimentConfig(
        model_slug="gemini-3-flash-preview",
        gpu="H100",
        triton_version="3.3.1",
        max_session_turns=7,
        aggregator="geomean",
        user_prompt_renderer=custom,
    )
    assert cfg.user_prompt_renderer(_sample_context(), cfg) == sentinel
    # System renderer unchanged — still the default adapter.
    assert cfg.system_prompt_renderer is default_system_prompt_renderer
