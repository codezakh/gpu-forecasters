"""Smoke tests for the native-abstain prompts, tool spec, and parser.

We do not exercise a real LLM here — the goal is to confirm that the
two prompts render with both tool names visible, that the JSON Schema
stays flat (Together gpt-oss compatibility), and that the parser
discriminates the predict and defer arms correctly.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import MagicMock

import pytest

from arid_badger.landscape_map.v2.abstain_estimator import (
    AbstainingLlmSpeedupEstimator,
    Deferral,
    Forecast,
)
from arid_badger.landscape_map.v2.abstain_prompt_rendering import (
    render_abstain_system_prompt,
    render_abstain_user_prompt,
)
from arid_badger.landscape_map.v2.abstain_tool_spec import (
    DEFER_TOOL_NAME,
    PREDICT_TOOL_NAME,
    both_openai_tool_specs,
    defer_parameters_schema,
)
from arid_badger.landscape_map.v2.domain import (
    HardwareContext,
    KernelImplementation,
    KernelRuntimeQuery,
    KernelTaskInfo,
)
from arid_badger.landscape_map.v2.parsing import EstimatorParseError


_HW = HardwareContext(
    device_name="A100-80GB",
    compute_capability=(8, 0),
    total_global_memory_gb=80.0,
    multiprocessor_count=108,
    max_threads_per_multiprocessor=2048,
    clock_rate_ghz=1.41,
    memory_clock_rate_ghz=1.215,
    memory_bus_width_bits=5120,
)


def _example_query() -> KernelRuntimeQuery:
    return KernelRuntimeQuery(
        task=KernelTaskInfo(op_name="trimul", level_id=2, task_id=0),
        reference=KernelImplementation(
            kernel_name="ref",
            code="def ref(): pass",
            runtime_ms=None,
        ),
        candidate=KernelImplementation(
            kernel_name="cand",
            code="// candidate cuda",
            runtime_ms=None,
        ),
        hardware=_HW,
    )


def test_defer_schema_is_flat() -> None:
    schema = defer_parameters_schema()
    assert "$defs" not in schema
    assert schema["properties"]["reason"]["type"] == "string"


def test_both_specs_have_distinct_names() -> None:
    specs = both_openai_tool_specs()
    names = {s["function"]["name"] for s in specs}
    assert names == {PREDICT_TOOL_NAME, DEFER_TOOL_NAME}


def test_system_prompt_mentions_both_tools_and_ten_factors() -> None:
    text = render_abstain_system_prompt()
    assert PREDICT_TOOL_NAME in text
    assert DEFER_TOOL_NAME in text
    # Ten-factor analysis guide must be preserved verbatim.
    for kw in [
        "Algorithmic complexity",
        "Memory access patterns",
        "Arithmetic intensity",
        "Thread divergence",
        "Occupancy",
        "Parallelism",
        "Synchronization",
        "launch overhead",
        "Library calls",
        "Data types",
    ]:
        assert kw in text, f"system prompt missing factor: {kw!r}"
    # Bin table preserved.
    for bin_name in [
        "SEVERE_SLOWDOWN",
        "MODERATE_SLOWDOWN",
        "MINOR_SPEEDUP",
        "EXTREME_SPEEDUP",
    ]:
        assert bin_name in text


def test_user_prompt_mentions_both_tools_and_hardware() -> None:
    text = render_abstain_user_prompt(_example_query())
    assert PREDICT_TOOL_NAME in text
    assert DEFER_TOOL_NAME in text
    assert "trimul" in text
    assert "A100-80GB" in text


def _mock_response(
    *,
    tool_calls: list[tuple[str, dict[str, Any]]],
    finish_reason: str = "tool_calls",
    content: str | None = None,
) -> Any:
    response = MagicMock()
    response.choices = [MagicMock()]
    response.choices[0].finish_reason = finish_reason
    msg = response.choices[0].message
    msg.content = content
    msg.tool_calls = []
    for name, args in tool_calls:
        tc = MagicMock()
        tc.function.name = name
        tc.function.arguments = json.dumps(args)
        msg.tool_calls.append(tc)
    response.usage = None
    return response


def test_parser_routes_predict_to_forecast() -> None:
    est = AbstainingLlmSpeedupEstimator(model_slug="dummy")
    response = _mock_response(
        tool_calls=[
            (
                PREDICT_TOOL_NAME,
                {
                    "predicted_bin": 5,
                    "p_severe_slowdown": 0.0,
                    "p_significant_slowdown": 0.0,
                    "p_moderate_slowdown": 0.0,
                    "p_minor_slowdown": 0.1,
                    "p_minor_speedup": 0.7,
                    "p_significant_speedup": 0.2,
                    "p_high_speedup": 0.0,
                    "p_extreme_speedup": 0.0,
                    "reasoning": "looks like a small win",
                },
            )
        ],
    )
    result, usage = est._parse_response(response)
    assert isinstance(result, Forecast)
    assert int(result.estimate.predicted_bin) == 5


def test_parser_routes_defer_to_deferral() -> None:
    est = AbstainingLlmSpeedupEstimator(model_slug="dummy")
    response = _mock_response(
        tool_calls=[(DEFER_TOOL_NAME, {"reason": "vendor library reference"})],
    )
    result, _usage = est._parse_response(response)
    assert isinstance(result, Deferral)
    assert "vendor" in result.reason


def test_parser_rejects_no_tool_call() -> None:
    est = AbstainingLlmSpeedupEstimator(model_slug="dummy")
    response = _mock_response(
        tool_calls=[], finish_reason="stop", content="thinking out loud"
    )
    with pytest.raises(EstimatorParseError, match="no tool"):
        est._parse_response(response)


def test_parser_rejects_multiple_tool_calls() -> None:
    est = AbstainingLlmSpeedupEstimator(model_slug="dummy")
    response = _mock_response(
        tool_calls=[
            (DEFER_TOOL_NAME, {"reason": "x"}),
            (DEFER_TOOL_NAME, {"reason": "y"}),
        ],
    )
    with pytest.raises(EstimatorParseError, match="multiple tools"):
        est._parse_response(response)
