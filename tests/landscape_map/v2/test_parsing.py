"""Tests for parse_tool_call_args: JSON parsing, renormalization, errors."""

from __future__ import annotations

import json
import math

import pytest

from gpu_forecasters.landscape_map.v2.domain import (
    SUCCESS_BINS,
    SpeedupBin,
)
from gpu_forecasters.landscape_map.v2.parsing import (
    EstimatorParseError,
    parse_tool_call_args,
)


def _wire_args(
    *,
    predicted_bin: int = 5,
    p_severe_slowdown: float = 0.0,
    p_significant_slowdown: float = 0.0,
    p_moderate_slowdown: float = 0.05,
    p_minor_slowdown: float = 0.1,
    p_minor_speedup: float = 0.6,
    p_significant_speedup: float = 0.2,
    p_high_speedup: float = 0.05,
    p_extreme_speedup: float = 0.0,
    reasoning: str = "fused ops, mild speedup expected",
) -> str:
    return json.dumps(
        {
            "predicted_bin": predicted_bin,
            "p_severe_slowdown": p_severe_slowdown,
            "p_significant_slowdown": p_significant_slowdown,
            "p_moderate_slowdown": p_moderate_slowdown,
            "p_minor_slowdown": p_minor_slowdown,
            "p_minor_speedup": p_minor_speedup,
            "p_significant_speedup": p_significant_speedup,
            "p_high_speedup": p_high_speedup,
            "p_extreme_speedup": p_extreme_speedup,
            "reasoning": reasoning,
        }
    )


def test_well_formed_unit_sum_round_trips() -> None:
    estimate = parse_tool_call_args(_wire_args())
    assert estimate.predicted_bin == SpeedupBin.MINOR_SPEEDUP
    assert math.isclose(estimate.raw_probability_sum, 1.0, abs_tol=1e-9)
    assert math.isclose(sum(estimate.bin_probabilities.values()), 1.0, abs_tol=1e-9)
    # bin 5 retains the dominant probability after renormalization
    assert estimate.bin_probabilities[SpeedupBin.MINOR_SPEEDUP] > 0.5


def test_undersummed_distribution_renormalizes() -> None:
    # Probabilities sum to 0.75; the renormalized distribution must
    # still sum to 1, and raw_probability_sum must record 0.75.
    estimate = parse_tool_call_args(
        _wire_args(
            p_severe_slowdown=0.0,
            p_significant_slowdown=0.0,
            p_moderate_slowdown=0.07,
            p_minor_slowdown=0.55,
            p_minor_speedup=0.10,
            p_significant_speedup=0.03,
            p_high_speedup=0.0,
            p_extreme_speedup=0.0,
        )
    )
    assert math.isclose(estimate.raw_probability_sum, 0.75, abs_tol=1e-9)
    assert math.isclose(sum(estimate.bin_probabilities.values()), 1.0, abs_tol=1e-9)


def test_oversummed_distribution_renormalizes() -> None:
    # Total = 1.10. After renormalization each value is scaled by 1/1.10.
    estimate = parse_tool_call_args(
        _wire_args(
            p_minor_speedup=0.60,
            p_significant_speedup=0.30,
            p_high_speedup=0.20,
            p_minor_slowdown=0.0,
            p_moderate_slowdown=0.0,
            p_extreme_speedup=0.0,
        )
    )
    assert math.isclose(estimate.raw_probability_sum, 1.10, abs_tol=1e-9)
    assert math.isclose(sum(estimate.bin_probabilities.values()), 1.0, abs_tol=1e-9)
    # Relative proportions preserved
    assert math.isclose(
        estimate.bin_probabilities[SpeedupBin.MINOR_SPEEDUP]
        / estimate.bin_probabilities[SpeedupBin.SIGNIFICANT_SPEEDUP],
        2.0,
    )


def test_all_zero_distribution_raises() -> None:
    args = _wire_args(
        p_severe_slowdown=0.0,
        p_significant_slowdown=0.0,
        p_moderate_slowdown=0.0,
        p_minor_slowdown=0.0,
        p_minor_speedup=0.0,
        p_significant_speedup=0.0,
        p_high_speedup=0.0,
        p_extreme_speedup=0.0,
    )
    with pytest.raises(EstimatorParseError, match="renormalize"):
        parse_tool_call_args(args)


def test_invalid_json_raises() -> None:
    with pytest.raises(EstimatorParseError, match="valid JSON"):
        parse_tool_call_args("{not json}")


def test_missing_field_raises() -> None:
    args = json.dumps({"predicted_bin": 5})
    with pytest.raises(EstimatorParseError, match="schema validation"):
        parse_tool_call_args(args)


def test_out_of_range_probability_raises() -> None:
    args = _wire_args(p_minor_speedup=1.5)
    with pytest.raises(EstimatorParseError, match="schema validation"):
        parse_tool_call_args(args)


def test_negative_probability_raises() -> None:
    args = _wire_args(p_minor_slowdown=-0.05)
    with pytest.raises(EstimatorParseError, match="schema validation"):
        parse_tool_call_args(args)


def test_invalid_predicted_bin_raises() -> None:
    args = _wire_args(predicted_bin=0)
    with pytest.raises(EstimatorParseError, match="schema validation"):
        parse_tool_call_args(args)


def test_blank_reasoning_raises() -> None:
    args = _wire_args(reasoning="")
    with pytest.raises(EstimatorParseError, match="schema validation"):
        parse_tool_call_args(args)


def test_estimate_covers_all_eight_bins() -> None:
    estimate = parse_tool_call_args(_wire_args())
    assert set(estimate.bin_probabilities.keys()) == set(SUCCESS_BINS)
