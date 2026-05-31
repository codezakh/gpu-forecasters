"""Smoke checks for the L3 problem loader.

Exercises the canonical-name registry and the lookup-by-name path
against the real KernelBench dataset on disk. The tests are fast (file
reads only), so they are not gated behind ``@pytest.mark.integration``.
"""

from __future__ import annotations

import pytest

from gpu_forecasters.kernelbench.v2.l3_problems import (
    TIER_B_PROBLEMS,
    L3ProblemReference,
    load_l3_problem,
    load_l3_problem_by_name,
)


def test_tier_b_registry_resolves_to_named_problems() -> None:
    """All five Tier-B references load with non-empty source code and
    the expected canonical filename stem."""
    assert len(TIER_B_PROBLEMS) == 5
    expected = {
        8: "8_ResNetBasicBlock",
        21: "21_EfficientNetMBConv",
        43: "43_MinGPTCausalAttention",
        44: "44_MiniGPTBlock",
        48: "48_Mamba2ReturnY",
    }
    by_id = {p.problem_id: p for p in TIER_B_PROBLEMS}
    assert by_id.keys() == expected.keys()
    for pid, name in expected.items():
        ref = by_id[pid]
        assert ref.name == name
        assert ref.reference_kernel_code.strip(), (
            f"Empty reference code for L3_{pid} ({name})"
        )
        assert "class Model" in ref.reference_kernel_code, (
            f"L3_{pid} reference code missing the canonical Model class"
        )


def test_load_by_id_and_name_agree() -> None:
    """``load_l3_problem(problem_id=8)`` and
    ``load_l3_problem_by_name("8_ResNetBasicBlock")`` return equivalent
    references."""
    by_id = load_l3_problem(problem_id=8)
    by_name = load_l3_problem_by_name("8_ResNetBasicBlock")
    assert by_id == by_name


def test_optional_problems_remain_loadable_off_registry() -> None:
    """L3_25 and L3_6 are deliberately excluded from TIER_B_PROBLEMS but
    must remain reachable via ``load_l3_problem(problem_id=...)`` for
    prompt-iteration spot-checks (gh070 spec)."""
    ref_25 = load_l3_problem(problem_id=25)
    ref_6 = load_l3_problem(problem_id=6)
    assert ref_25.name.startswith("25_")
    assert ref_6.name.startswith("6_")
    assert all(
        ref.problem_id not in {6, 25} for ref in TIER_B_PROBLEMS
    )


def test_unknown_id_raises() -> None:
    """A problem id outside L3 surfaces an error rather than returning
    a sentinel — caller errors must fail fast."""
    with pytest.raises(Exception):  # KernelBench raises a KeyError-like
        _ = load_l3_problem(problem_id=10_000)


def test_unknown_name_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        _ = load_l3_problem_by_name("not_a_real_problem")


def test_reference_is_frozen() -> None:
    """``L3ProblemReference`` is immutable so it can be safely shared
    across the run-loop closure boundary."""
    ref = TIER_B_PROBLEMS[0]
    assert isinstance(ref, L3ProblemReference)
    with pytest.raises(Exception):
        ref.problem_id = 999  # pyright: ignore[reportAttributeAccessIssue]
