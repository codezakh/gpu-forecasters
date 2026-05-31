"""Render the native-abstain system + user prompts from Jinja templates.

The templates ``prompts/abstain_system.j2`` and
``prompts/abstain_user.j2`` are byte-for-byte aligned with the predict-
only v2 templates except that:

* the "Output Format" section names two tools instead of one;
* a new "Predict or Defer" section sits between "How to Analyze" and
  "Output Format", describing when deferral is appropriate;
* the user prompt closes with "call exactly one of the two tools"
  instead of "call the tool exactly once."

Everything else — the ten-factor analysis guide, the bin table, the
hardware-context block, the speedup-definition section — lifts from
the predict-only prompt verbatim.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from gpu_forecasters.landscape_map.v2.abstain_tool_spec import (
    DEFER_TOOL_NAME,
    PREDICT_TOOL_NAME,
)
from gpu_forecasters.landscape_map.v2.domain import KernelRuntimeQuery


_PROMPTS_DIR = Path(__file__).parent / "prompts"
_JINJA_ENV = Environment(
    loader=FileSystemLoader(str(_PROMPTS_DIR)),
    autoescape=False,
)


def render_abstain_system_prompt() -> str:
    return _JINJA_ENV.get_template("abstain_system.j2").render(
        predict_tool_name=PREDICT_TOOL_NAME,
        defer_tool_name=DEFER_TOOL_NAME,
    )


def render_abstain_user_prompt(query: KernelRuntimeQuery) -> str:
    return _JINJA_ENV.get_template("abstain_user.j2").render(
        task=query.task,
        reference=query.reference,
        candidate=query.candidate,
        hardware=query.hardware,
        predict_tool_name=PREDICT_TOOL_NAME,
        defer_tool_name=DEFER_TOOL_NAME,
    )


__all__ = [
    "render_abstain_system_prompt",
    "render_abstain_user_prompt",
]
