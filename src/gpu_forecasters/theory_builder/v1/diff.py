"""SEARCH/REPLACE diff parsing + application for the markdown world model.

The format the LLM emits — adopted because LLMs are already fluent in
it (aider's edit format):

    <<<<<<< SEARCH
    ...exact text to find...
    =======
    ...replacement text...
    >>>>>>> REPLACE

Multiple blocks may appear in one response. ``parse_diff_blocks``
extracts them out of free-form LLM output; ``apply_diffs`` applies them
in order against the document.

Application rules (deliberately strict — fuzzy matching produces
silent corruption):

* If ``search`` is empty, ``replace`` is appended to the document
  separated by a blank line. This is how new sections get seeded.
* Otherwise ``search`` must match exactly once. Zero matches or
  multiple matches → ``DiffApplyError``.
* Empty ``replace`` is fine; that's a deletion.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from arid_badger.theory_builder.v1.domain import WorldModelDiff


class DiffApplyError(ValueError):
    """Raised when a diff block can't be applied — usually because
    ``search`` matched zero or multiple locations in the document."""


_SEARCH_REPLACE_RE = re.compile(
    r"<{3,}\s*SEARCH\s*\n(.*?)={3,}\s*\n(.*?)>{3,}\s*REPLACE",
    re.DOTALL,
)


def parse_diff_blocks(text: str) -> list[WorldModelDiff]:
    """Extract every SEARCH/REPLACE block from free-form LLM output.

    Returns the list in source order. Blocks with no match are simply
    omitted; the caller decides whether that's acceptable.

    Search/replace bodies are stripped of a single trailing newline if
    present — the closing fence (``=======`` / ``>>>>>>> REPLACE``)
    appears on its own line, so the captured body always ends with a
    newline that's a fence delimiter, not content."""
    blocks: list[WorldModelDiff] = []
    for m in _SEARCH_REPLACE_RE.finditer(text):
        search = _strip_trailing_newline(m.group(1))
        replace = _strip_trailing_newline(m.group(2))
        blocks.append(WorldModelDiff(search=search, replace=replace))
    return blocks


def _strip_trailing_newline(body: str) -> str:
    if body.endswith("\n"):
        return body[:-1]
    return body


@dataclass(frozen=True)
class DiffApplyOutcome:
    """Result of applying a sequence of diffs.

    ``applied`` is the new document. ``per_block_outcomes`` mirrors the
    input order — each entry is either ``None`` for a successful apply
    or an error string for a failed one. The applier short-circuits on
    the first failure; entries past the failure are not present."""

    applied: str
    per_block_outcomes: list[str | None]


def apply_diffs(text: str, diffs: list[WorldModelDiff]) -> str:
    """Apply each diff in order. Raises ``DiffApplyError`` on the first
    failure with a message naming the offending block.

    The caller (the builder's retry loop) uses the exception message
    to feed back to the LLM so it can correct the malformed diff on
    the next attempt."""
    current = text
    for i, diff in enumerate(diffs):
        try:
            current = _apply_one(current, diff)
        except DiffApplyError as exc:
            raise DiffApplyError(f"diff {i}: {exc}") from exc
    return current


def _apply_one(text: str, diff: WorldModelDiff) -> str:
    if diff.search == "":
        # Append-mode: seed a new section. Separate from prior content
        # with a blank line iff the document already has content.
        if not text:
            return diff.replace
        if text.endswith("\n\n"):
            return text + diff.replace
        if text.endswith("\n"):
            return text + "\n" + diff.replace
        return text + "\n\n" + diff.replace
    occurrences = text.count(diff.search)
    if occurrences == 0:
        raise DiffApplyError(
            "SEARCH block did not match any location in the document"
        )
    if occurrences > 1:
        raise DiffApplyError(
            f"SEARCH block matched {occurrences} locations; must be unique"
        )
    return text.replace(diff.search, diff.replace, 1)


__all__ = [
    "DiffApplyError",
    "DiffApplyOutcome",
    "apply_diffs",
    "parse_diff_blocks",
]
