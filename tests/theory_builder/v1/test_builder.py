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


def _make_hypothesis() -> Hypothesis:
    return Hypothesis(
        bottleneck="b",
        intervention="i",
        prediction="p",
        code_references=[],
    )


def _empty_result(hypothesis_id: ULID) -> ExperimentResult[NoFeedback]:
    return ExperimentResult[NoFeedback](
        hypothesis_id=hypothesis_id, trials=[]
    )


_EXPLANATION_TAGS = """\
<gap>g</gap>
<mechanism>m</mechanism>
<belief_update>u</belief_update>
"""


def test_propose_explanation_first_turn_done_with_no_edits():
    """LLM emits the three tags + <done/> in turn 1 with no diffs.
    World model is returned unchanged; ``Explanation.diffs`` is empty."""
    h = _make_hypothesis()
    response = _EXPLANATION_TAGS + "\n<done/>\n"
    stub = _ScriptedAcompletion([response])
    with patch("litellm.acompletion", stub):
        builder = _make_builder()
        wm = WorldModel(
            kernel_description="trimul", text="## Beliefs\n- one\n"
        )
        explanation, new_text = builder.propose_explanation(
            wm, h, _empty_result(h.id)
        )
    assert new_text == "## Beliefs\n- one\n"
    assert explanation.diffs == []
    assert len(stub.calls) == 1


def test_propose_explanation_iterative_multi_edit_happy_path():
    """Two edits applied across two turns, terminated by <done/> on
    turn 3. The second turn's user prompt must include the document
    state *after* the first edit applied."""
    h = _make_hypothesis()
    turn1 = _EXPLANATION_TAGS + """
<<<<<<< SEARCH
- one
=======
- one (revised)
>>>>>>> REPLACE
"""
    turn2 = """
<<<<<<< SEARCH
=======
- two
>>>>>>> REPLACE
"""
    turn3 = "<done/>"
    stub = _ScriptedAcompletion([turn1, turn2, turn3])
    with patch("litellm.acompletion", stub):
        builder = _make_builder()
        wm = WorldModel(
            kernel_description="trimul", text="## Beliefs\n- one\n"
        )
        explanation, new_text = builder.propose_explanation(
            wm, h, _empty_result(h.id)
        )
    assert "- one (revised)" in new_text
    assert "- two" in new_text
    assert len(explanation.diffs) == 2
    assert len(stub.calls) == 3
    # Turn 2's user prompt must contain the post-turn-1 document.
    turn2_messages = stub.calls[1]
    last_user_turn2 = turn2_messages[-1]
    assert last_user_turn2["role"] == "user"
    assert "Edit applied" in last_user_turn2["content"]
    assert "- one (revised)" in last_user_turn2["content"]
    # Turn 3 must include the post-turn-2 document.
    last_user_turn3 = stub.calls[2][-1]
    assert "- two" in last_user_turn3["content"]


def test_propose_explanation_recovers_from_apply_failure():
    """A SEARCH-mismatch on turn 1 is fed back to the LLM with the
    error and the (still-unchanged) current document; a corrected
    edit on turn 2 then applies, and <done/> in the same response
    terminates the loop."""
    h = _make_hypothesis()
    bad_then_good = _EXPLANATION_TAGS + """
<<<<<<< SEARCH
nonexistent line
=======
replacement
>>>>>>> REPLACE
"""
    good_with_done = """
<<<<<<< SEARCH
=======
- new entry
>>>>>>> REPLACE
<done/>
"""
    stub = _ScriptedAcompletion([bad_then_good, good_with_done])
    with patch("litellm.acompletion", stub):
        builder = _make_builder()
        wm = WorldModel(
            kernel_description="trimul",
            text="## Beliefs\n- one\n",
        )
        explanation, new_text = builder.propose_explanation(
            wm, h, _empty_result(h.id)
        )
    assert "new entry" in new_text
    assert "## Beliefs" in new_text
    assert len(stub.calls) == 2
    # Turn 2 must name the apply failure AND include the current doc
    # so the LLM can re-target its SEARCH.
    last_user = stub.calls[1][-1]
    assert "could not be applied" in last_user["content"]
    assert "## Beliefs" in last_user["content"]
    assert len(explanation.diffs) == 1


def test_propose_explanation_exhausts_apply_failures():
    """Three consecutive bad SEARCHes blow the apply-failure budget
    even though the turn budget is larger."""
    h = _make_hypothesis()
    bad = _EXPLANATION_TAGS + """
<<<<<<< SEARCH
nonexistent
=======
x
>>>>>>> REPLACE
"""
    bad_subsequent = """
<<<<<<< SEARCH
also nonexistent
=======
x
>>>>>>> REPLACE
"""
    stub = _ScriptedAcompletion(
        [bad, bad_subsequent, bad_subsequent, bad_subsequent]
    )
    with patch("litellm.acompletion", stub):
        builder = LLMWorldModelBuilder[NoFeedback](
            model_slug="test/model",
            result_renderer=_NoOpRenderer(),
            max_turns=10,
            max_apply_failures=3,
        )
        wm = WorldModel(
            kernel_description="trimul", text="## Beliefs\n- one\n"
        )
        with pytest.raises(BuilderError, match="explanation"):
            _ = builder.propose_explanation(wm, h, _empty_result(h.id))
    assert len(stub.calls) == 3


def test_propose_explanation_exhausts_turns_when_never_done():
    """LLM keeps applying edits and never says <done/>. Turn budget
    runs out and the builder raises."""
    h = _make_hypothesis()
    turn1 = _EXPLANATION_TAGS + """
<<<<<<< SEARCH
=======
- a
>>>>>>> REPLACE
"""
    later = """
<<<<<<< SEARCH
=======
- a
>>>>>>> REPLACE
"""
    stub = _ScriptedAcompletion([turn1] + [later] * 10)
    with patch("litellm.acompletion", stub):
        builder = LLMWorldModelBuilder[NoFeedback](
            model_slug="test/model",
            result_renderer=_NoOpRenderer(),
            max_turns=3,
        )
        wm = WorldModel(
            kernel_description="trimul", text="## Beliefs\n- one\n"
        )
        with pytest.raises(BuilderError, match="3 turn"):
            _ = builder.propose_explanation(wm, h, _empty_result(h.id))
    assert len(stub.calls) == 3


def test_propose_explanation_recovers_from_missing_tags_on_turn_1():
    """Turn 1 lacks the explanation tags. The builder feeds the parse
    error back; turn 2 supplies the tags + <done/>."""
    h = _make_hypothesis()
    no_tags = "I'm thinking out loud but forgot the tags."
    with_tags = _EXPLANATION_TAGS + "<done/>\n"
    stub = _ScriptedAcompletion([no_tags, with_tags])
    with patch("litellm.acompletion", stub):
        builder = _make_builder()
        wm = WorldModel(
            kernel_description="trimul", text="## Beliefs\n- one\n"
        )
        explanation, new_text = builder.propose_explanation(
            wm, h, _empty_result(h.id)
        )
    assert explanation.gap == "g"
    assert new_text == "## Beliefs\n- one\n"
    assert len(stub.calls) == 2
    last_user = stub.calls[1][-1]
    assert "missing required tags" in last_user["content"]


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
    response = _EXPLANATION_TAGS + "<done/>\n"
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
