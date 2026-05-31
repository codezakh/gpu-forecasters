"""Tagged-span parsing of LLM responses into domain objects.

The LLM is asked to think aloud and then emit each domain field inside
a named tag, e.g. ``<bottleneck>...</bottleneck>``. This pattern is
robust on both frontier and open models — structured-output APIs are
brittle on open-source LLMs.

Each helper raises ``ParseError`` with a concrete reason that the
builder feeds back to the LLM on retry.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from arid_badger.theory_builder.v1.diff import parse_diff_blocks
from arid_badger.theory_builder.v1.domain import (
    Explanation,
    Hypothesis,
    HypothesisStatus,
    WorldModelDiff,
)
from ulid import ULID


class ParseError(ValueError):
    """Raised when a response is missing a required tagged span."""


_TAG_RE_CACHE: dict[str, re.Pattern[str]] = {}


def _tag_re(name: str) -> re.Pattern[str]:
    if name not in _TAG_RE_CACHE:
        _TAG_RE_CACHE[name] = re.compile(
            rf"<{name}>(.*?)</{name}>", re.DOTALL
        )
    return _TAG_RE_CACHE[name]


def extract_tag(text: str, name: str) -> str:
    """Extract the contents of the LAST occurrence of ``<name>...</name>``.

    The "last" choice mirrors how the LLM is prompted: it may think
    aloud and rewrite a draft before its final answer; the trailing
    occurrence is authoritative."""
    matches = _tag_re(name).findall(text)
    if not matches:
        raise ParseError(f"missing required tag <{name}>...</{name}>")
    return matches[-1].strip()


def extract_optional_tag(text: str, name: str) -> str | None:
    matches = _tag_re(name).findall(text)
    if not matches:
        return None
    return matches[-1].strip()


def extract_list_tag(text: str, name: str, item_name: str) -> list[str]:
    """Extract each ``<item_name>...</item_name>`` inside the last
    ``<name>...</name>`` block. Returns ``[]`` if ``<name>`` is missing
    or has no items."""
    container = extract_optional_tag(text, name)
    if container is None:
        return []
    items = _tag_re(item_name).findall(container)
    return [item.strip() for item in items]


@dataclass(frozen=True)
class ParsedHypothesis:
    """Intermediate parse result before promotion to a ``Hypothesis``
    domain object. ``id`` is assigned by the builder on success."""

    bottleneck: str
    intervention: str
    prediction: str
    code_references: list[str]


def parse_hypothesis(response: str) -> ParsedHypothesis:
    """Pull a hypothesis out of an LLM response.

    Required tags: ``<bottleneck>``, ``<intervention>``, ``<prediction>``.
    Optional: ``<code_references>`` containing one or more
    ``<reference>...</reference>`` entries."""
    bottleneck = extract_tag(response, "bottleneck")
    intervention = extract_tag(response, "intervention")
    prediction = extract_tag(response, "prediction")
    references = extract_list_tag(response, "code_references", "reference")
    if not bottleneck:
        raise ParseError("<bottleneck> was empty")
    if not intervention:
        raise ParseError("<intervention> was empty")
    if not prediction:
        raise ParseError("<prediction> was empty")
    return ParsedHypothesis(
        bottleneck=bottleneck,
        intervention=intervention,
        prediction=prediction,
        code_references=references,
    )


def parse_hypothesis_into_domain(response: str) -> Hypothesis:
    parsed = parse_hypothesis(response)
    return Hypothesis(
        id=ULID(),
        bottleneck=parsed.bottleneck,
        intervention=parsed.intervention,
        prediction=parsed.prediction,
        code_references=parsed.code_references,
        status="open",
    )


_VALID_STATUSES: tuple[HypothesisStatus, ...] = (
    "open",
    "under_investigation",
    "closed",
    "established",
)


def parse_explanation(response: str, hypothesis_id: ULID) -> Explanation:
    """Pull an explanation + diffs out of an LLM response.

    Required tags: ``<gap>``, ``<mechanism>``, ``<belief_update>``.
    Diff blocks (``<<<<<<< SEARCH ... ======= ... >>>>>>> REPLACE``)
    are parsed by ``diff.parse_diff_blocks``; zero diffs is allowed.
    """
    tags = parse_explanation_tags(response)
    diffs: list[WorldModelDiff] = parse_diff_blocks(response)
    return Explanation(
        hypothesis_id=hypothesis_id,
        gap=tags.gap,
        mechanism=tags.mechanism,
        belief_update=tags.belief_update,
        diffs=diffs,
    )


@dataclass(frozen=True)
class ParsedExplanationTags:
    """The three explanation tags only — no diffs.

    Used by the iterative-editing builder, which collects diffs across
    multiple turns rather than parsing them out of a single response."""

    gap: str
    mechanism: str
    belief_update: str


def parse_explanation_tags(response: str) -> ParsedExplanationTags:
    """Pull the three required explanation tags out of a response.

    Raises ``ParseError`` if any tag is missing or empty. Diff blocks
    in the response are ignored — that's the iterative builder's job
    to handle one turn at a time."""
    gap = extract_tag(response, "gap")
    mechanism = extract_tag(response, "mechanism")
    belief_update = extract_tag(response, "belief_update")
    if not gap:
        raise ParseError("<gap> was empty")
    if not mechanism:
        raise ParseError("<mechanism> was empty")
    if not belief_update:
        raise ParseError("<belief_update> was empty")
    return ParsedExplanationTags(
        gap=gap, mechanism=mechanism, belief_update=belief_update
    )


# Permissive: <done/>, <done />, or <done></done>; case-insensitive.
_DONE_RE = re.compile(r"<\s*done\s*/?\s*>", re.IGNORECASE)


def has_done_signal(response: str) -> bool:
    """True if ``response`` contains a ``<done/>``-style termination
    signal. The iterative explanation loop treats this as "I have no
    further edits to emit; commit what's been applied."""
    return _DONE_RE.search(response) is not None


__all__ = [
    "ParseError",
    "ParsedHypothesis",
    "ParsedExplanationTags",
    "extract_tag",
    "extract_optional_tag",
    "extract_list_tag",
    "parse_hypothesis",
    "parse_hypothesis_into_domain",
    "parse_explanation",
    "parse_explanation_tags",
    "has_done_signal",
]
