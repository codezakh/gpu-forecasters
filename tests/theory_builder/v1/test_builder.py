"""LLM-backed builder retry behaviour.

Patches ``litellm.acompletion`` with an async stub returning scripted
responses so we can exercise the parse/apply retry loop without an
external service. The point is to confirm that a malformed first
response is followed by a *new* prompt whose final user turn names
the apply error, and that a valid response on the second try
succeeds.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from ulid import ULID

from arid_badger.hill_climbing.domain import NoFeedback
from arid_badger.theory_builder.v1.builder import (
    BuilderError,
    LLMWorldModelBuilder,
)
from arid_badger.theory_builder.v1.domain import (
    ExperimentResult,
    Hypothesis,
    WorldModel,
)


class _StubChoice:
    def __init__(self, content: str) -> None:
        self.message = type("M", (), {"content": content})()


class _StubResponse:
    def __init__(self, content: str) -> None:
        self.choices = [_StubChoice(content)]


class _ScriptedAcompletion:
    """Async stub: each ``__call__`` returns the next scripted
    response. Records the final messages payload for inspection."""

    def __init__(self, responses: list[str]) -> None:
        self.responses = list(responses)
        self.calls: list[list[dict[str, str]]] = []

    async def __call__(self, **kwargs: Any) -> _StubResponse:
        self.calls.append(list(kwargs["messages"]))
        return _StubResponse(self.responses.pop(0))


class _NoOpRenderer:
    """Stand-in for ``ExperimentResultRenderer`` during a hypothesis-
    only test."""

    def render(self, result: ExperimentResult[NoFeedback]) -> str:
        return "(no result rendered)"


def _make_builder(
    *, max_retries: int = 2
) -> LLMWorldModelBuilder[NoFeedback]:
    return LLMWorldModelBuilder[NoFeedback](
        model_slug="test/model",
        result_renderer=_NoOpRenderer(),
        max_retries=max_retries,
    )


def test_propose_hypothesis_happy_path():
    response = """\
<bottleneck>x</bottleneck>
<intervention>y</intervention>
<prediction>z</prediction>
"""
    stub = _ScriptedAcompletion([response])
    with patch("litellm.acompletion", stub):
        builder = _make_builder()
        h = builder.propose_hypothesis(
            WorldModel(kernel_description="trimul")
        )
    assert h.bottleneck == "x"
    assert len(stub.calls) == 1


def test_propose_hypothesis_retries_on_parse_error():
    bad = "no tags at all"
    good = """\
<bottleneck>x</bottleneck>
<intervention>y</intervention>
<prediction>z</prediction>
"""
    stub = _ScriptedAcompletion([bad, good])
    with patch("litellm.acompletion", stub):
        builder = _make_builder()
        h = builder.propose_hypothesis(
            WorldModel(kernel_description="trimul")
        )
    assert h.bottleneck == "x"
    assert len(stub.calls) == 2
    # Second call's messages must include the failed assistant turn
    # plus a user turn naming the parse error.
    second_messages = stub.calls[1]
    assert any(
        m["role"] == "assistant" and m["content"] == bad
        for m in second_messages
    )
    last_user_turn = second_messages[-1]
    assert last_user_turn["role"] == "user"
    assert "could not be parsed" in last_user_turn["content"]


def test_propose_hypothesis_exhausts_retries():
    bad = "still no tags"
    stub = _ScriptedAcompletion([bad, bad, bad])
    with patch("litellm.acompletion", stub):
        builder = _make_builder(max_retries=2)
        with pytest.raises(BuilderError, match="hypothesis"):
            _ = builder.propose_hypothesis(
                WorldModel(kernel_description="trimul")
            )
    assert len(stub.calls) == 3  # initial + 2 retries


def test_propose_explanation_retries_on_diff_apply_error():
    """First response references a SEARCH that doesn't exist in the
    world model; second response uses an empty SEARCH (append)."""
    h = Hypothesis(
        bottleneck="b",
        intervention="i",
        prediction="p",
        code_references=[],
    )
    bad = """\
<gap>g</gap>
<mechanism>m</mechanism>
<belief_update>u</belief_update>

<<<<<<< SEARCH
nonexistent line
=======
replacement
>>>>>>> REPLACE
"""
    good = """\
<gap>g</gap>
<mechanism>m</mechanism>
<belief_update>u</belief_update>

<<<<<<< SEARCH
=======
- new entry
>>>>>>> REPLACE
"""
    stub = _ScriptedAcompletion([bad, good])
    with patch("litellm.acompletion", stub):
        builder = _make_builder()
        wm = WorldModel(
            kernel_description="trimul",
            text="## Beliefs\n- one\n",
        )
        result: ExperimentResult[NoFeedback] = ExperimentResult[
            NoFeedback
        ](hypothesis_id=h.id, trials=[])
        explanation, new_text = builder.propose_explanation(
            wm, h, result
        )
    assert explanation.hypothesis_id == h.id
    assert "new entry" in new_text
    assert "## Beliefs" in new_text
    assert len(stub.calls) == 2
    # The second-call user turn must name the diff-apply failure.
    last_user_turn = stub.calls[1][-1]
    assert "could not be applied" in last_user_turn["content"]


def test_propose_explanation_passes_hypothesis_id_through():
    """``Explanation.hypothesis_id`` is set by the builder, not parsed
    out of the LLM response — make sure it survives."""
    hid = ULID()
    h = Hypothesis(
        id=hid,
        bottleneck="b",
        intervention="i",
        prediction="p",
        code_references=[],
    )
    response = """\
<gap>g</gap>
<mechanism>m</mechanism>
<belief_update>u</belief_update>
"""
    stub = _ScriptedAcompletion([response])
    with patch("litellm.acompletion", stub):
        builder = _make_builder()
        explanation, _ = builder.propose_explanation(
            WorldModel(kernel_description="trimul"),
            h,
            ExperimentResult[NoFeedback](
                hypothesis_id=hid, trials=[]
            ),
        )
    assert explanation.hypothesis_id == hid
