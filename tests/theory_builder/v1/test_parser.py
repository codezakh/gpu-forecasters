"""Tagged-span parsing of LLM responses."""

from __future__ import annotations

import pytest
from ulid import ULID

from gpu_forecasters.theory_builder.v1.parser import (
    ParseError,
    extract_list_tag,
    extract_optional_tag,
    extract_tag,
    parse_explanation,
    parse_hypothesis,
    parse_hypothesis_into_domain,
)


def test_extract_tag_returns_last_occurrence():
    text = "<x>first</x>\nthought aloud\n<x>final</x>"
    assert extract_tag(text, "x") == "final"


def test_extract_tag_missing_raises():
    with pytest.raises(ParseError, match="missing"):
        _ = extract_tag("nothing here", "x")


def test_extract_optional_tag_returns_none_when_missing():
    assert extract_optional_tag("nothing", "x") is None


def test_extract_list_tag_returns_items():
    text = """\
<refs>
  <r>foo</r>
  <r>bar baz</r>
</refs>"""
    assert extract_list_tag(text, "refs", "r") == ["foo", "bar baz"]


def test_extract_list_tag_missing_returns_empty():
    assert extract_list_tag("nothing", "refs", "r") == []


def test_parse_hypothesis_happy_path():
    response = """\
let me think about this.

<bottleneck>The kernel is bound by global memory bandwidth at line 42.</bottleneck>
<intervention>Switch tile size from 32 to 64 to halve the number of loads.</intervention>
<prediction>Geomean speedup rises by >=20%.</prediction>
<code_references>
  <reference>line 42: tl.load with stride 1</reference>
  <reference>BLOCK_M=32</reference>
</code_references>
"""
    parsed = parse_hypothesis(response)
    assert parsed.bottleneck.startswith("The kernel is bound")
    assert parsed.intervention.startswith("Switch tile size")
    assert parsed.prediction.startswith("Geomean speedup")
    assert parsed.code_references == [
        "line 42: tl.load with stride 1",
        "BLOCK_M=32",
    ]


def test_parse_hypothesis_into_domain_assigns_id_and_status():
    response = """\
<bottleneck>x</bottleneck>
<intervention>y</intervention>
<prediction>z</prediction>
"""
    h = parse_hypothesis_into_domain(response)
    assert h.status == "open"
    assert isinstance(h.id, ULID)
    assert h.code_references == []


def test_parse_hypothesis_missing_field_raises():
    response = """\
<bottleneck>x</bottleneck>
<intervention>y</intervention>
"""
    with pytest.raises(ParseError, match="prediction"):
        _ = parse_hypothesis(response)


def test_parse_explanation_happy_path():
    hid = ULID()
    response = """\
<gap>The kernel got 1.5x as predicted but seqlen=1024 lagged.</gap>
<mechanism>The new tile size hits an L2 capacity wall above seqlen=512.</mechanism>
<belief_update>Tile=64 helps small seqlen but not large; need to investigate L2 footprint.</belief_update>

<<<<<<< SEARCH
- H-1: tile=64 may help.
=======
- H-1: tile=64 helps below seqlen=512; above that, L2 footprint dominates. (closed)
>>>>>>> REPLACE
"""
    expl = parse_explanation(response, hid)
    assert expl.hypothesis_id == hid
    assert expl.gap.startswith("The kernel got")
    assert len(expl.diffs) == 1
    assert "L2 footprint" in expl.diffs[0].replace


def test_parse_explanation_zero_diffs_allowed():
    hid = ULID()
    response = """\
<gap>Result confirmed prediction exactly.</gap>
<mechanism>The intervention worked as expected.</mechanism>
<belief_update>No update needed; world model already captures this.</belief_update>
"""
    expl = parse_explanation(response, hid)
    assert expl.diffs == []
