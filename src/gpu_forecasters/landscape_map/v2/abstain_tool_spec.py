"""The two-tool surface for the native-abstain v2 surrogate.

The model is given exactly two tools and must call one (and only one):

* :func:`predict_tool_spec` — the existing
  ``submit_kernel_runtime_estimate`` from
  :mod:`gpu_forecasters.landscape_map.v2.tool_spec`.
* :func:`defer_tool_spec` — a new ``defer_to_real_evaluator`` tool
  whose only argument is a one-to-two-sentence ``reason``.

We keep both tools flat (no nested submodels, no $defs/$ref) so the
spec is compatible with Together gpt-oss tool-call validation, matching
the contract the predict-only spec already meets.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from gpu_forecasters.landscape_map.v2.tool_spec import (
    TOOL_DESCRIPTION as PREDICT_TOOL_DESCRIPTION,
    TOOL_NAME as PREDICT_TOOL_NAME,
    parameters_schema as predict_parameters_schema,
)


DEFER_TOOL_NAME = "defer_to_real_evaluator"
DEFER_TOOL_DESCRIPTION = (
    "Defer this candidate to a real GPU evaluator instead of predicting "
    "its speedup bin. Use this only when you are too uncertain to "
    "produce a reasoned distribution. Call this exactly once instead "
    "of submit_kernel_runtime_estimate, never both."
)


class DeferArguments(BaseModel, frozen=True):
    """Wire-format arguments for the defer tool.

    A single ``reason`` field carries the LLM's rationale. We keep the
    tool intentionally narrow so callers do not have to validate
    multiple fields and so the model has nothing else to fill in
    beyond the abstention rationale.
    """

    reason: str = Field(
        min_length=1,
        description=(
            "1-2 sentences naming the specific source of uncertainty "
            "that motivates deferral."
        ),
    )


def defer_parameters_schema() -> dict[str, Any]:
    """JSON Schema for the defer tool's arguments."""
    schema = DeferArguments.model_json_schema()
    assert "$defs" not in schema, (
        "DeferArguments JSON Schema gained $defs; the model must stay "
        "flat for compatibility with Together gpt-oss tool calling."
    )
    return schema


def predict_tool_spec_openai() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": PREDICT_TOOL_NAME,
            "description": PREDICT_TOOL_DESCRIPTION,
            "parameters": predict_parameters_schema(),
        },
    }


def defer_tool_spec_openai() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": DEFER_TOOL_NAME,
            "description": DEFER_TOOL_DESCRIPTION,
            "parameters": defer_parameters_schema(),
        },
    }


def both_openai_tool_specs() -> list[dict[str, Any]]:
    """Pair of tool specs for ``litellm.completion(..., tools=[...])``.

    Order is (predict, defer); the model is free to call either.
    """
    return [predict_tool_spec_openai(), defer_tool_spec_openai()]


def predict_cookbook_tool_spec() -> dict[str, Any]:
    """Predict tool spec in the cookbook ``ToolSpec`` shape (renderer-compatible)."""
    return {
        "name": PREDICT_TOOL_NAME,
        "description": PREDICT_TOOL_DESCRIPTION,
        "parameters": predict_parameters_schema(),
    }


def defer_cookbook_tool_spec() -> dict[str, Any]:
    """Defer tool spec in the cookbook ``ToolSpec`` shape (renderer-compatible)."""
    return {
        "name": DEFER_TOOL_NAME,
        "description": DEFER_TOOL_DESCRIPTION,
        "parameters": defer_parameters_schema(),
    }


def both_cookbook_tool_specs() -> list[dict[str, Any]]:
    """Pair of cookbook-format tool specs for renderer-driven sampling.

    Order is (predict, defer); used by Tinker-backed estimators and the
    abstain RL env to register both tools on the conversation prefix.
    """
    return [predict_cookbook_tool_spec(), defer_cookbook_tool_spec()]


__all__ = [
    "DEFER_TOOL_DESCRIPTION",
    "DEFER_TOOL_NAME",
    "DeferArguments",
    "PREDICT_TOOL_DESCRIPTION",
    "PREDICT_TOOL_NAME",
    "both_cookbook_tool_specs",
    "both_openai_tool_specs",
    "defer_cookbook_tool_spec",
    "defer_parameters_schema",
    "defer_tool_spec_openai",
    "predict_cookbook_tool_spec",
    "predict_tool_spec_openai",
]
