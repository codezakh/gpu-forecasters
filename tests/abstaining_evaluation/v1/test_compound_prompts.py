"""Unit tests for the compound mutation prompt formatters.

The forecast-arm prompt's exact wording is kept loose — these tests
check that all the structural pieces (parent code, predicted bin,
distribution, reasoning, closing instruction) appear, not the exact
prose. The real-eval-arm prompt's structure mirrors gpu_mode_kernel's
formatter at fork time; we cover its four feedback arms with the
same shape of assertions.
"""

from __future__ import annotations

import pytest

from gpu_forecasters.abstaining_evaluation.v1.observation import ForecastObservation
from gpu_forecasters.abstaining_evaluation.v1.prompts import (
    format_forecast_feedback_prompt,
    format_real_eval_feedback_prompt,
)
from gpu_forecasters.gpu_mode_kernel.core import (
    CompileFailedFeedback,
    IncorrectFeedback,
    RuntimeErrorFeedback,
    SuccessFeedback,
)
from gpu_forecasters.gpu_mode_kernel.packs.trimul import TriMulCaseSpeedup
from gpu_forecasters.landscape_map.v2 import (
    SUCCESS_BINS,
    KernelRuntimeEstimate,
    SpeedupBin,
)


_BASE = "[BASE PROMPT BODY]"
_PARENT = "def custom_kernel(): pass"


def _trimul_case() -> TriMulCaseSpeedup:
    return TriMulCaseSpeedup(
        seqlen=256,
        bs=2,
        dim=128,
        hiddendim=128,
        nomask=True,
        distribution="normal",
        speedup=1.42,
        runtime_ns=1000.0,
        ref_runtime_ns=1420.0,
    )


def _estimate_for_bin(predicted: SpeedupBin) -> KernelRuntimeEstimate:
    """Concentrate probability mass on ``predicted`` to make assertions clean."""
    probs = {b: 0.0 for b in SUCCESS_BINS}
    probs[predicted] = 1.0
    return KernelRuntimeEstimate(
        predicted_bin=predicted,
        bin_probabilities=probs,
        reasoning="reasoning string for tests",
        raw_probability_sum=1.0,
    )


# ---------------------------------------------------------------------------
# Forecast-arm formatter
# ---------------------------------------------------------------------------


def test_forecast_prompt_includes_all_structural_pieces() -> None:
    forecast = ForecastObservation(
        estimate=_estimate_for_bin(SpeedupBin.HIGH_SPEEDUP),
        expected_speedup=2.5,
    )
    prompt = format_forecast_feedback_prompt(
        base_prompt=_BASE, parent_code=_PARENT, forecast=forecast
    )
    assert _BASE in prompt
    assert _PARENT in prompt
    # Marks itself as a forecast, not a measurement.
    assert "SURROGATE PREDICTOR" in prompt
    assert "no GPU run" in prompt
    # Predicted bin label appears.
    assert SpeedupBin.HIGH_SPEEDUP.label in prompt
    # Expected speedup is rendered.
    assert "2.500x" in prompt
    # Reasoning appears.
    assert "reasoning string for tests" in prompt
    # Closing rewrite instruction.
    assert "rewrite the entire kernel" in prompt


def test_forecast_prompt_renders_full_distribution() -> None:
    """Every one of the eight success bins must appear with its
    probability in the rendered distribution block."""
    probs = {b: 0.0 for b in SUCCESS_BINS}
    probs[SpeedupBin.MINOR_SPEEDUP] = 0.6
    probs[SpeedupBin.SIGNIFICANT_SPEEDUP] = 0.4
    estimate = KernelRuntimeEstimate(
        predicted_bin=SpeedupBin.MINOR_SPEEDUP,
        bin_probabilities=probs,
        reasoning="r",
        raw_probability_sum=1.0,
    )
    forecast = ForecastObservation(estimate=estimate, expected_speedup=1.5)
    prompt = format_forecast_feedback_prompt(
        base_prompt=_BASE, parent_code=_PARENT, forecast=forecast
    )
    for bin_ in SUCCESS_BINS:
        # Each bin appears with its label.
        assert bin_.label in prompt, f"{bin_.label} missing from prompt"
    # Two non-zero probabilities are rendered.
    assert "0.600" in prompt
    assert "0.400" in prompt


def test_forecast_prompt_rejects_empty_inputs() -> None:
    forecast = ForecastObservation(
        estimate=_estimate_for_bin(SpeedupBin.MINOR_SPEEDUP),
        expected_speedup=1.0,
    )
    with pytest.raises(ValueError, match="base_prompt"):
        _ = format_forecast_feedback_prompt(
            base_prompt="", parent_code=_PARENT, forecast=forecast
        )
    with pytest.raises(ValueError, match="parent_code"):
        _ = format_forecast_feedback_prompt(
            base_prompt=_BASE, parent_code="", forecast=forecast
        )


# ---------------------------------------------------------------------------
# Real-eval-arm formatter (sanity coverage of all four arms)
# ---------------------------------------------------------------------------


def test_real_eval_prompt_renders_compile_failed() -> None:
    fb = CompileFailedFeedback(compilation_error="SyntaxError: invalid syntax")
    prompt = format_real_eval_feedback_prompt(
        base_prompt=_BASE, parent_code=_PARENT, feedback=fb
    )
    assert _BASE in prompt
    assert _PARENT in prompt
    assert "failed to compile" in prompt
    assert "SyntaxError" in prompt


def test_real_eval_prompt_renders_runtime_error() -> None:
    fb = RuntimeErrorFeedback(
        runtime_error_name="ValueError",
        runtime_error="something went wrong",
        traceback="File foo.py line 1 in bar()",
    )
    prompt = format_real_eval_feedback_prompt(
        base_prompt=_BASE, parent_code=_PARENT, feedback=fb
    )
    assert "raised an exception at runtime" in prompt
    assert "ValueError" in prompt
    assert "something went wrong" in prompt
    assert "File foo.py" in prompt


def test_real_eval_prompt_renders_incorrect() -> None:
    fb = IncorrectFeedback(error_message="output norm differs by 1e-2")
    prompt = format_real_eval_feedback_prompt(
        base_prompt=_BASE, parent_code=_PARENT, feedback=fb
    )
    assert "incorrect output compared to the reference" in prompt
    assert "output norm differs by 1e-2" in prompt


def test_real_eval_prompt_renders_success_with_per_case_breakdown() -> None:
    fb = SuccessFeedback[TriMulCaseSpeedup](
        aggregated_speedup=1.42,
        aggregation_method="geomean",
        per_case_speedups=[_trimul_case()],
    )
    prompt = format_real_eval_feedback_prompt(
        base_prompt=_BASE, parent_code=_PARENT, feedback=fb
    )
    assert "Your kernel is correct" in prompt
    assert "Aggregated speedup: 1.420x" in prompt
    assert "geomean" in prompt
    assert "Per-case breakdown" in prompt
    # The case's format_for_prompt() output must appear.
    assert "seqlen=256" in prompt


def test_real_eval_prompt_rejects_empty_inputs() -> None:
    fb = IncorrectFeedback(error_message="x")
    with pytest.raises(ValueError, match="base_prompt"):
        _ = format_real_eval_feedback_prompt(
            base_prompt="", parent_code=_PARENT, feedback=fb
        )
    with pytest.raises(ValueError, match="parent_code"):
        _ = format_real_eval_feedback_prompt(
            base_prompt=_BASE, parent_code="", feedback=fb
        )
