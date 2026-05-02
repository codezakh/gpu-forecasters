"""Jinja-based prompt rendering for the v2 surrogate.

The templates in ``prompts/`` mirror the v1 ``landscape_map`` prompts
verbatim except that:
  - the v1 "Confidence Scale" section (Likert) is replaced with a
    "Probability Distribution" section describing the simplex over
    bins 1..8;
  - the v1 "Output Format" section (XML + fenced JSON) is replaced
    with an instruction to call the v2 tool exactly once.

Everything else — the bin table, the ten-factor analysis guide, and
the hardware-context table — is byte-for-byte aligned with v1, so
the v2 surrogate sees the same conditioning information v1 did.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from .domain import KernelRuntimeQuery
from .tool_spec import TOOL_NAME


_PROMPTS_DIR = Path(__file__).parent / "prompts"
_JINJA_ENV = Environment(
    loader=FileSystemLoader(str(_PROMPTS_DIR)),
    autoescape=False,
)


def render_system_prompt() -> str:
    return _JINJA_ENV.get_template("system.j2").render(tool_name=TOOL_NAME)


def render_user_prompt(query: KernelRuntimeQuery) -> str:
    return _JINJA_ENV.get_template("user.j2").render(
        task=query.task,
        reference=query.reference,
        candidate=query.candidate,
        hardware=query.hardware,
        tool_name=TOOL_NAME,
    )
