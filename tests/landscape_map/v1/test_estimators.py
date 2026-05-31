"""
Tests for StubEstimator confidence distribution logic and LLM response parsing pipeline.
"""

import json
import textwrap

import pytest

from gpu_forecasters.landscape_map.v1.domain import (
    KernelImplementation,
    KernelRuntimeQuery,
    KernelTaskInfo,
    LikertConfidence,
    SpeedupBin,
)
from gpu_forecasters.landscape_map.v1.llm_estimator import (
    EstimatorParseError,
    _extract_json_from_response,
    _parse_llm_response,
    _resolve_bin_key,
    _resolve_confidence,
)
from gpu_forecasters.landscape_map.v1.stub_estimator import StubEstimator


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_query() -> KernelRuntimeQuery:
    return KernelRuntimeQuery(
        task=KernelTaskInfo(op_name="vector_add", level_id=1, task_id=1),
        reference=KernelImplementation(kernel_name="ref", code="pass", runtime_ms=1.0),
        candidate=KernelImplementation(kernel_name="cand", code="pass", runtime_ms=1.0),
    )


# ---------------------------------------------------------------------------
# StubEstimator — confidence distribution logic
# ---------------------------------------------------------------------------

def test_stub_estimator_confidence_distribution() -> None:
    """Default stub assigns VERY_HIGH to its fixed bin and VERY_LOW to all others.
    FAILURE (bin 0) must be excluded from the confidence dict entirely.
    """
    estimator = StubEstimator()
    estimate, usage = estimator.estimate(_make_query())

    assert estimate.predicted_bin == SpeedupBin.MINOR_SLOWDOWN
    assert estimate.bin_confidences[SpeedupBin.MINOR_SLOWDOWN] == LikertConfidence.VERY_HIGH
    # Spot-check: a non-fixed bin should be VERY_LOW
    assert estimate.bin_confidences[SpeedupBin.MINOR_SPEEDUP] == LikertConfidence.VERY_LOW
    # FAILURE must not appear in bin_confidences
    assert SpeedupBin.FAILURE not in estimate.bin_confidences
    # Must cover exactly bins 1-8
    assert len(estimate.bin_confidences) == 8
    assert usage is None


def test_stub_estimator_custom_bin() -> None:
    """VERY_HIGH confidence shifts to whichever bin is passed to the constructor."""
    estimator = StubEstimator(fixed_bin=SpeedupBin.EXTREME_SPEEDUP)
    estimate, _ = estimator.estimate(_make_query())

    assert estimate.predicted_bin == SpeedupBin.EXTREME_SPEEDUP
    assert estimate.bin_confidences[SpeedupBin.EXTREME_SPEEDUP] == LikertConfidence.VERY_HIGH
    # The previously-default bin must now be VERY_LOW
    assert estimate.bin_confidences[SpeedupBin.MINOR_SLOWDOWN] == LikertConfidence.VERY_LOW


# ---------------------------------------------------------------------------
# _extract_json_from_response — 4-strategy extraction cascade
# ---------------------------------------------------------------------------

def test_extract_json_prefers_kernel_runtime_block() -> None:
    """Strategy 1 (kernel-runtime-estimate block) wins over strategy 2 (json block)
    when both are present. The two blocks contain different predicted_bin values
    so we can tell which one was parsed.
    """
    content = textwrap.dedent("""\
        ```kernel-runtime-estimate
        {"predicted_bin": 5}
        ```

        ```json
        {"predicted_bin": 3}
        ```
    """)
    result = _extract_json_from_response(content)
    assert result["predicted_bin"] == 5


def test_extract_json_falls_through_to_json_block() -> None:
    """When there is no kernel-runtime-estimate block, the json block is used."""
    content = textwrap.dedent("""\
        Some analysis here.

        ```json
        {"predicted_bin": 6, "bin_confidences": {}}
        ```
    """)
    result = _extract_json_from_response(content)
    assert result["predicted_bin"] == 6


def test_extract_json_falls_through_to_raw_json() -> None:
    """When there are no fenced blocks, bare JSON anywhere in the response is used."""
    content = 'Here is the result: {"predicted_bin": 7, "reasoning": "fast"}'
    result = _extract_json_from_response(content)
    assert result["predicted_bin"] == 7


def test_extract_json_raises_on_no_json() -> None:
    """A response with no parseable JSON raises EstimatorParseError."""
    with pytest.raises(EstimatorParseError):
        _extract_json_from_response("No JSON here, just text.")


# ---------------------------------------------------------------------------
# _parse_llm_response — full response parsing pipeline
# ---------------------------------------------------------------------------

def _make_response(predicted_bin: int, confidences: dict, reasoning: str = "test") -> str:
    payload = json.dumps({
        "predicted_bin": predicted_bin,
        "bin_confidences": confidences,
        "reasoning": reasoning,
    })
    return f"```kernel-runtime-estimate\n{payload}\n```"


def test_parse_llm_response_complete() -> None:
    """A well-formed response produces the correct predicted_bin, all 8 bins in
    bin_confidences, and the reasoning string from the JSON field.
    """
    confidences = {str(i): "very_low" for i in range(1, 9)}
    confidences["5"] = "high"
    response = _make_response(predicted_bin=5, confidences=confidences, reasoning="candidate fuses ops")

    result = _parse_llm_response(response)

    assert result.predicted_bin == SpeedupBin.MINOR_SPEEDUP
    assert len(result.bin_confidences) == 8
    assert result.bin_confidences[SpeedupBin.MINOR_SPEEDUP] == LikertConfidence.HIGH
    assert result.reasoning == "candidate fuses ops"


def test_parse_llm_response_fills_missing_confidences() -> None:
    """Bins omitted from the LLM response are filled with VERY_LOW by _parse_bin_confidences."""
    # Only provide confidences for bins 4, 5, 6
    confidences = {"4": "moderate", "5": "high", "6": "moderate"}
    response = _make_response(predicted_bin=5, confidences=confidences)

    result = _parse_llm_response(response)

    # The three provided bins keep their values
    assert result.bin_confidences[SpeedupBin.MINOR_SLOWDOWN] == LikertConfidence.MODERATE
    assert result.bin_confidences[SpeedupBin.MINOR_SPEEDUP] == LikertConfidence.HIGH
    # All missing bins must be filled with VERY_LOW
    for missing_bin in [
        SpeedupBin.SEVERE_SLOWDOWN,
        SpeedupBin.SIGNIFICANT_SLOWDOWN,
        SpeedupBin.MODERATE_SLOWDOWN,
        SpeedupBin.HIGH_SPEEDUP,
        SpeedupBin.EXTREME_SPEEDUP,
    ]:
        assert result.bin_confidences[missing_bin] == LikertConfidence.VERY_LOW
    # Total must still be 8 (bins 1-8)
    assert len(result.bin_confidences) == 8


def test_parse_llm_response_accepts_bin_names_as_keys() -> None:
    """_resolve_bin_key accepts SpeedupBin names (case-insensitive) in addition to ints."""
    confidences = {"minor_speedup": "high", "EXTREME_SPEEDUP": "very_low"}
    response = _make_response(predicted_bin=5, confidences=confidences)

    result = _parse_llm_response(response)

    assert result.bin_confidences[SpeedupBin.MINOR_SPEEDUP] == LikertConfidence.HIGH
    assert result.bin_confidences[SpeedupBin.EXTREME_SPEEDUP] == LikertConfidence.VERY_LOW


def test_parse_llm_response_reasoning_fallback_to_xml() -> None:
    """When the JSON reasoning field is empty, the <runtimeCalculation> XML tag is used."""
    payload = json.dumps({
        "predicted_bin": 5,
        "bin_confidences": {},
        "reasoning": "",  # empty -> triggers XML fallback
    })
    content = (
        "<runtimeCalculation>analysis here</runtimeCalculation>\n"
        f"```kernel-runtime-estimate\n{payload}\n```"
    )
    result = _parse_llm_response(content)
    assert result.reasoning == "analysis here"


def test_parse_llm_response_missing_predicted_bin_raises() -> None:
    """JSON that lacks a predicted_bin key raises EstimatorParseError."""
    payload = json.dumps({"bin_confidences": {"5": "high"}, "reasoning": "test"})
    content = f"```kernel-runtime-estimate\n{payload}\n```"

    with pytest.raises(EstimatorParseError, match="predicted_bin"):
        _parse_llm_response(content)


# ---------------------------------------------------------------------------
# _resolve_bin_key and _resolve_confidence — key normalization
# ---------------------------------------------------------------------------

def test_resolve_bin_key_by_integer_string() -> None:
    assert _resolve_bin_key("5") == SpeedupBin.MINOR_SPEEDUP


def test_resolve_bin_key_by_name_case_insensitive() -> None:
    assert _resolve_bin_key("minor_speedup") == SpeedupBin.MINOR_SPEEDUP
    assert _resolve_bin_key("MINOR_SPEEDUP") == SpeedupBin.MINOR_SPEEDUP


def test_resolve_confidence_case_insensitive() -> None:
    assert _resolve_confidence("Very_High") == LikertConfidence.VERY_HIGH
    assert _resolve_confidence("very_high") == LikertConfidence.VERY_HIGH
