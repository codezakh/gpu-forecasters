"""SEARCH/REPLACE diff parsing + application."""

from __future__ import annotations

import pytest

from gpu_forecasters.theory_builder.v1.diff import (
    DiffApplyError,
    apply_diffs,
    parse_diff_blocks,
)
from gpu_forecasters.theory_builder.v1.domain import WorldModelDiff


def test_parse_single_block():
    text = """\
some preamble

<<<<<<< SEARCH
old text
=======
new text
>>>>>>> REPLACE

trailing chatter
"""
    blocks = parse_diff_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].search == "old text"
    assert blocks[0].replace == "new text"


def test_parse_multiple_blocks_in_order():
    text = """\
<<<<<<< SEARCH
A
=======
A'
>>>>>>> REPLACE

<<<<<<< SEARCH
B
=======
B'
>>>>>>> REPLACE
"""
    blocks = parse_diff_blocks(text)
    assert [b.search for b in blocks] == ["A", "B"]
    assert [b.replace for b in blocks] == ["A'", "B'"]


def test_parse_multiline_search():
    text = """\
<<<<<<< SEARCH
line one
line two
=======
new content
>>>>>>> REPLACE
"""
    blocks = parse_diff_blocks(text)
    assert len(blocks) == 1
    assert blocks[0].search == "line one\nline two"


def test_parse_no_blocks_returns_empty():
    assert parse_diff_blocks("just chatter, no diff blocks") == []


def test_apply_single_block_unique_match():
    doc = "alpha\nbeta\ngamma\n"
    diffs = [WorldModelDiff(search="beta", replace="BETA")]
    assert apply_diffs(doc, diffs) == "alpha\nBETA\ngamma\n"


def test_apply_zero_match_raises():
    doc = "alpha\nbeta\n"
    diffs = [WorldModelDiff(search="zeta", replace="X")]
    with pytest.raises(DiffApplyError, match="did not match"):
        _ = apply_diffs(doc, diffs)


def test_apply_multiple_match_raises():
    doc = "x\nx\n"
    diffs = [WorldModelDiff(search="x", replace="Y")]
    with pytest.raises(DiffApplyError, match="matched 2"):
        _ = apply_diffs(doc, diffs)


def test_apply_empty_search_appends():
    doc = "## Existing\n- one\n"
    diffs = [
        WorldModelDiff(search="", replace="## New section\n- two\n")
    ]
    result = apply_diffs(doc, diffs)
    assert result.startswith(doc)
    assert "## New section" in result
    assert "- two" in result


def test_apply_empty_search_into_empty_doc():
    diffs = [WorldModelDiff(search="", replace="## first section\n")]
    assert apply_diffs("", diffs) == "## first section\n"


def test_apply_in_order():
    doc = "A\nB\n"
    diffs = [
        WorldModelDiff(search="A", replace="C"),
        WorldModelDiff(search="C", replace="D"),
    ]
    assert apply_diffs(doc, diffs) == "D\nB\n"


def test_apply_first_failure_short_circuits():
    doc = "A\n"
    diffs = [
        WorldModelDiff(search="missing", replace="X"),
        WorldModelDiff(search="A", replace="Y"),  # would succeed
    ]
    with pytest.raises(DiffApplyError, match="diff 0"):
        _ = apply_diffs(doc, diffs)
