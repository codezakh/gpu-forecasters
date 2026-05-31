"""Parsing tests for the Tinker-backed abstain estimator.

We don't drive a real Tinker SamplingClient — we exercise
``_parse_tokens`` directly by monkeypatching the renderer's
``parse_response`` to return hand-built assistant messages, mirroring
the approach in ``test_rl_env.py``.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from tinker_cookbook.renderers import Message, ToolCall, get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

from gpu_forecasters.landscape_map.v2 import (
    Deferral,
    Forecast,
    SpeedupBin,
    TinkerSamplingClientAbstainingEstimator,
)
from gpu_forecasters.landscape_map.v2.parsing import EstimatorParseError


pytest.importorskip("transformers")


def _make_estimator() -> TinkerSamplingClientAbstainingEstimator:
    tokenizer = get_tokenizer("openai/gpt-oss-20b")
    renderer = get_renderer("gpt_oss_medium_reasoning", tokenizer=tokenizer)
    # Inject a mock sampling_client so the constructor doesn't try to
    # talk to Tinker. The tests don't call sample_async; they exercise
    # _parse_tokens directly.
    return TinkerSamplingClientAbstainingEstimator(
        sampling_client=MagicMock(),
        renderer=renderer,
    )


def _predict_message() -> Message:
    args = (
        '{"predicted_bin": 5, '
        '"p_severe_slowdown": 0.01, "p_significant_slowdown": 0.04, '
        '"p_moderate_slowdown": 0.05, "p_minor_slowdown": 0.10, '
        '"p_minor_speedup": 0.50, "p_significant_speedup": 0.20, '
        '"p_high_speedup": 0.05, "p_extreme_speedup": 0.05, '
        '"reasoning": "test"}'
    )
    return Message(
        role="assistant",
        content="",
        tool_calls=[
            ToolCall(
                function=ToolCall.FunctionBody(
                    name="submit_kernel_runtime_estimate", arguments=args
                )
            )
        ],
    )


def _defer_message(reason: str = "novel hardware shape") -> Message:
    return Message(
        role="assistant",
        content="",
        tool_calls=[
            ToolCall(
                function=ToolCall.FunctionBody(
                    name="defer_to_real_evaluator",
                    arguments='{"reason": "' + reason + '"}',
                )
            )
        ],
    )


def test_parse_predict_yields_forecast() -> None:
    estimator = _make_estimator()
    estimator._renderer.parse_response = lambda tokens: (_predict_message(), True)  # type: ignore[method-assign]
    outcome, usage = estimator._parse_tokens([1, 2, 3], prompt_length=42)
    assert isinstance(outcome, Forecast)
    assert outcome.estimate.predicted_bin == SpeedupBin.MINOR_SPEEDUP
    assert usage is not None
    assert usage.input_tokens == 42
    assert usage.output_tokens == 3


def test_parse_defer_yields_deferral() -> None:
    estimator = _make_estimator()
    estimator._renderer.parse_response = lambda tokens: (  # type: ignore[method-assign]
        _defer_message(reason="too uncertain"),
        True,
    )
    outcome, usage = estimator._parse_tokens([1, 2, 3, 4], prompt_length=10)
    assert isinstance(outcome, Deferral)
    assert outcome.reason == "too uncertain"
    assert usage is not None
    assert usage.output_tokens == 4


def test_parse_no_tool_call_raises() -> None:
    estimator = _make_estimator()
    msg = Message(role="assistant", content="I don't know.")
    estimator._renderer.parse_response = lambda tokens: (msg, True)  # type: ignore[method-assign]
    with pytest.raises(EstimatorParseError, match="parsed no tool_calls"):
        estimator._parse_tokens([1], prompt_length=5)


def test_parse_multiple_tool_calls_raises() -> None:
    estimator = _make_estimator()
    args = (
        '{"predicted_bin": 5, '
        '"p_severe_slowdown": 0.01, "p_significant_slowdown": 0.04, '
        '"p_moderate_slowdown": 0.05, "p_minor_slowdown": 0.10, '
        '"p_minor_speedup": 0.50, "p_significant_speedup": 0.20, '
        '"p_high_speedup": 0.05, "p_extreme_speedup": 0.05, '
        '"reasoning": "x"}'
    )
    both = Message(
        role="assistant",
        content="",
        tool_calls=[
            ToolCall(
                function=ToolCall.FunctionBody(
                    name="submit_kernel_runtime_estimate", arguments=args
                )
            ),
            ToolCall(
                function=ToolCall.FunctionBody(
                    name="defer_to_real_evaluator",
                    arguments='{"reason": "noisy"}',
                )
            ),
        ],
    )
    estimator._renderer.parse_response = lambda tokens: (both, True)  # type: ignore[method-assign]
    with pytest.raises(EstimatorParseError, match="multiple tools"):
        estimator._parse_tokens([1], prompt_length=5)


def test_parse_unknown_tool_raises() -> None:
    estimator = _make_estimator()
    weird = Message(
        role="assistant",
        content="",
        tool_calls=[
            ToolCall(
                function=ToolCall.FunctionBody(
                    name="some_other_tool", arguments="{}"
                )
            ),
        ],
    )
    estimator._renderer.parse_response = lambda tokens: (weird, True)  # type: ignore[method-assign]
    with pytest.raises(EstimatorParseError, match="unexpected tool"):
        estimator._parse_tokens([1], prompt_length=5)


def test_parse_malformed_predict_args_raises() -> None:
    estimator = _make_estimator()
    bad = Message(
        role="assistant",
        content="",
        tool_calls=[
            ToolCall(
                function=ToolCall.FunctionBody(
                    name="submit_kernel_runtime_estimate",
                    arguments="not json",
                )
            ),
        ],
    )
    estimator._renderer.parse_response = lambda tokens: (bad, True)  # type: ignore[method-assign]
    with pytest.raises(EstimatorParseError):
        estimator._parse_tokens([1], prompt_length=5)


def test_parse_malformed_defer_args_raises() -> None:
    estimator = _make_estimator()
    bad = Message(
        role="assistant",
        content="",
        tool_calls=[
            ToolCall(
                function=ToolCall.FunctionBody(
                    name="defer_to_real_evaluator",
                    arguments="{}",  # missing required `reason`
                )
            ),
        ],
    )
    estimator._renderer.parse_response = lambda tokens: (bad, True)  # type: ignore[method-assign]
    with pytest.raises(EstimatorParseError, match="defer tool arguments"):
        estimator._parse_tokens([1], prompt_length=5)
