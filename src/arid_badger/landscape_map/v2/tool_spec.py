"""Single source of truth for the surrogate's one tool.

The model is asked to call ``submit_kernel_runtime_estimate`` exactly
once with arguments that match :class:`SubmitEstimateArguments`. We
intentionally keep the wire-format model *flat* (eight named ``p_*``
fields, no nested submodel) so that the generated JSON Schema has no
``$defs`` / ``$ref`` — Together's gpt-oss tool-call validator (and
some other providers) reject schemas with refs.

The wire-format model is distinct from the domain
:class:`KernelRuntimeEstimate` in ``domain.py``: the wire model carries
the eight separate floats *unrenormalized* as the LLM produced them,
and :func:`parse_tool_call_args` in ``parsing.py`` translates from the
wire model to the domain model (which uses a renormalized
``dict[SpeedupBin, float]`` keyed by bin enum values).
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from .domain import SUCCESS_BINS, SpeedupBin


TOOL_NAME = "submit_kernel_runtime_estimate"
TOOL_DESCRIPTION = (
    "Submit your prediction of the candidate kernel's speedup relative to "
    "the reference. Call this exactly once, after you have finished "
    "reasoning."
)


class SubmitEstimateArguments(BaseModel, frozen=True):
    """Wire-format arguments the model emits via the tool call.

    The eight ``p_*`` fields are the per-bin probabilities the model
    assigned. They are validated to be in ``[0, 1]`` here, but their
    *sum* is not required to equal 1 at this layer — the parsing
    function in ``parsing.py`` renormalizes and surfaces the original
    sum to the domain model.
    """

    predicted_bin: Literal[1, 2, 3, 4, 5, 6, 7, 8] = Field(
        description=(
            "Most likely speedup bin (1=severe slowdown, 8=extreme speedup)."
        )
    )
    p_severe_slowdown: float = Field(
        ge=0.0, le=1.0, description="P(speedup <= 0.25x)"
    )
    p_significant_slowdown: float = Field(
        ge=0.0, le=1.0, description="P(0.25x < S <= 0.5x)"
    )
    p_moderate_slowdown: float = Field(
        ge=0.0, le=1.0, description="P(0.5x < S <= 0.71x)"
    )
    p_minor_slowdown: float = Field(
        ge=0.0, le=1.0, description="P(0.71x < S <= 1.0x)"
    )
    p_minor_speedup: float = Field(
        ge=0.0, le=1.0, description="P(1.0x < S <= 1.41x)"
    )
    p_significant_speedup: float = Field(
        ge=0.0, le=1.0, description="P(1.41x < S <= 2.0x)"
    )
    p_high_speedup: float = Field(
        ge=0.0, le=1.0, description="P(2.0x < S <= 4.0x)"
    )
    p_extreme_speedup: float = Field(
        ge=0.0, le=1.0, description="P(speedup > 4.0x)"
    )
    reasoning: str = Field(
        min_length=1,
        description="Concise 1-3 sentence rationale for the prediction.",
    )

    def per_bin_dict(self) -> dict[SpeedupBin, float]:
        """Bin-keyed view of the eight raw probabilities (unrenormalized)."""
        return {
            SpeedupBin.SEVERE_SLOWDOWN: self.p_severe_slowdown,
            SpeedupBin.SIGNIFICANT_SLOWDOWN: self.p_significant_slowdown,
            SpeedupBin.MODERATE_SLOWDOWN: self.p_moderate_slowdown,
            SpeedupBin.MINOR_SLOWDOWN: self.p_minor_slowdown,
            SpeedupBin.MINOR_SPEEDUP: self.p_minor_speedup,
            SpeedupBin.SIGNIFICANT_SPEEDUP: self.p_significant_speedup,
            SpeedupBin.HIGH_SPEEDUP: self.p_high_speedup,
            SpeedupBin.EXTREME_SPEEDUP: self.p_extreme_speedup,
        }


def parameters_schema() -> dict[str, Any]:
    """JSON Schema for the tool's arguments."""
    schema = SubmitEstimateArguments.model_json_schema()
    # Sanity-check: no $defs/$ref. If this fails the model has gained a
    # nested submodel by accident and providers like Together will reject.
    assert "$defs" not in schema, (
        "SubmitEstimateArguments JSON Schema gained $defs; the model must "
        "stay flat for compatibility with Together gpt-oss tool calling."
    )
    return schema


def openai_tool_spec() -> dict[str, Any]:
    """Tool spec in the OpenAI ``tools=[...]`` shape (LiteLLM-compatible)."""
    return {
        "type": "function",
        "function": {
            "name": TOOL_NAME,
            "description": TOOL_DESCRIPTION,
            "parameters": parameters_schema(),
        },
    }


def cookbook_tool_spec() -> dict[str, Any]:
    """Tool spec in the cookbook ``ToolSpec`` shape (renderer-compatible)."""
    return {
        "name": TOOL_NAME,
        "description": TOOL_DESCRIPTION,
        "parameters": parameters_schema(),
    }


__all__ = [
    "SUCCESS_BINS",  # convenience re-export
    "SubmitEstimateArguments",
    "TOOL_DESCRIPTION",
    "TOOL_NAME",
    "cookbook_tool_spec",
    "openai_tool_spec",
    "parameters_schema",
]
