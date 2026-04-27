"""Tests for the kernel-agnostic mutation prompt assembly.

Covers:
- ``build_base_prompt`` substitutes ``gpu_name`` / ``triton_version``
  into the rules block and concatenates onto the pack's description.
- ``format_feedback_prompt`` shapes each of the four
  ``KernelExecutionFeedback`` arms differently — error truncation,
  per-case sort order on the success arm, structural framing.
- ``extract_last_python_codeblock`` picks the last block, strips the
  fence, and returns ``None`` on no-match.
"""

from __future__ import annotations

import pytest
from pydantic import ConfigDict
from typing_extensions import TypedDict

from arid_badger.gpu_mode_kernel.core import (
    CaseSpeedupBase,
    CompileFailedFeedback,
    IncorrectFeedback,
    RuntimeErrorFeedback,
    SuccessFeedback,
)
from arid_badger.gpu_mode_kernel.kernel_pack import KernelPack
from arid_badger.gpu_mode_kernel.prompts import (
    build_base_prompt,
    extract_last_python_codeblock,
    format_feedback_prompt,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class FixtureTestArgs(TypedDict):
    shape: int


class FixtureCaseSpeedup(CaseSpeedupBase):
    model_config = ConfigDict(frozen=True)

    shape: int

    def format_for_prompt(self) -> str:
        return f"shape={self.shape}: {self.speedup:.3f}x"


def _fixture_pack() -> KernelPack[FixtureTestArgs, FixtureCaseSpeedup]:
    return KernelPack(
        name="fixture",
        modal_app_name="fixture-app",
        correctness_cases=[],
        benchmark_cases=[],
        ref_kernel=lambda data: data,
        generate_input=lambda **kwargs: kwargs,
        check_implementation=lambda data, output: (True, ""),
        seed_kernel_code="",
        determinism_ctx=None,
        case_speedup_type=FixtureCaseSpeedup,
        kernel_description_body="DESCRIPTION_OF_KERNEL\nSecond line.",
    )


# ---------------------------------------------------------------------------
# build_base_prompt
# ---------------------------------------------------------------------------


def test_build_base_prompt_includes_description_and_rules() -> None:
    pack = _fixture_pack()
    prompt = build_base_prompt(
        pack, gpu_name="A100-80GB", triton_version="3.3.1"
    )
    assert "DESCRIPTION_OF_KERNEL" in prompt
    assert "Rules:" in prompt
    assert "Nvidia A100-80GB" in prompt
    assert "triton 3.3.1" in prompt


def test_build_base_prompt_substitutes_gpu_and_triton() -> None:
    pack = _fixture_pack()
    a = build_base_prompt(pack, gpu_name="L40S", triton_version="2.3.0")
    b = build_base_prompt(pack, gpu_name="A100-80GB", triton_version="3.3.1")
    assert a != b
    assert "L40S" in a and "L40S" not in b
    assert "2.3.0" in a and "2.3.0" not in b


# ---------------------------------------------------------------------------
# extract_last_python_codeblock
# ---------------------------------------------------------------------------


def test_extract_picks_last_block() -> None:
    text = (
        "Here is a draft:\n"
        "```python\nx = 1\n```\n\n"
        "Actually, the final version is:\n"
        "```python\ny = 2\n```\n"
    )
    code = extract_last_python_codeblock(text)
    assert code == "y = 2"


def test_extract_strips_trailing_fence() -> None:
    text = "```python\ndef foo(): pass\n```"
    code = extract_last_python_codeblock(text)
    assert code == "def foo(): pass"


def test_extract_returns_none_on_no_match() -> None:
    assert extract_last_python_codeblock("no code here") is None
    assert extract_last_python_codeblock("") is None


def test_extract_returns_none_on_empty_block() -> None:
    text = "```python\n```"
    assert extract_last_python_codeblock(text) is None


# ---------------------------------------------------------------------------
# format_feedback_prompt — each arm
# ---------------------------------------------------------------------------


_BASE = "BASE_PROMPT"
_PARENT = "def custom_kernel(data):\n    return data"


def test_compile_failed_arm() -> None:
    feedback = CompileFailedFeedback(compilation_error="SyntaxError on line 3")
    prompt = format_feedback_prompt(
        base_prompt=_BASE,
        previous_kernel_code=_PARENT,
        feedback=feedback,
    )
    assert prompt.startswith(_BASE)
    assert _PARENT in prompt
    assert "failed to compile" in prompt
    assert "SyntaxError on line 3" in prompt


def test_runtime_error_arm_includes_traceback() -> None:
    feedback = RuntimeErrorFeedback(
        runtime_error_name="RuntimeError",
        runtime_error="oops",
        traceback="frame1\nframe2\nframe3",
    )
    prompt = format_feedback_prompt(
        base_prompt=_BASE, previous_kernel_code=_PARENT, feedback=feedback
    )
    assert "RuntimeError" in prompt
    assert "oops" in prompt
    assert "frame3" in prompt


def test_incorrect_arm() -> None:
    feedback = IncorrectFeedback(error_message="elements differ at idx 5")
    prompt = format_feedback_prompt(
        base_prompt=_BASE, previous_kernel_code=_PARENT, feedback=feedback
    )
    assert "incorrect" in prompt
    assert "elements differ at idx 5" in prompt


def test_success_arm_sorts_slowest_first_and_uses_pack_formatter() -> None:
    cases = [
        FixtureCaseSpeedup(
            shape=100, speedup=4.0, runtime_ns=1_000.0, ref_runtime_ns=4_000.0
        ),
        FixtureCaseSpeedup(
            shape=200, speedup=1.5, runtime_ns=2_000.0, ref_runtime_ns=3_000.0
        ),
        FixtureCaseSpeedup(
            shape=300, speedup=2.0, runtime_ns=1_500.0, ref_runtime_ns=3_000.0
        ),
    ]
    feedback = SuccessFeedback[FixtureCaseSpeedup](
        aggregated_speedup=2.5,
        aggregation_method="geomean",
        per_case_speedups=cases,
    )
    prompt = format_feedback_prompt(
        base_prompt=_BASE, previous_kernel_code=_PARENT, feedback=feedback
    )
    # Pack-supplied formatter is what places the per-case lines.
    assert "shape=200: 1.500x" in prompt
    assert "shape=300: 2.000x" in prompt
    assert "shape=100: 4.000x" in prompt
    # Sort: slowest (1.5x) listed before fastest (4.0x).
    idx_slow = prompt.index("shape=200: 1.500x")
    idx_fast = prompt.index("shape=100: 4.000x")
    assert idx_slow < idx_fast


def test_empty_base_prompt_raises() -> None:
    with pytest.raises(ValueError, match="base_prompt"):
        format_feedback_prompt(
            base_prompt="",
            previous_kernel_code=_PARENT,
            feedback=IncorrectFeedback(error_message="x"),
        )


def test_empty_parent_code_raises() -> None:
    with pytest.raises(ValueError, match="previous_kernel_code"):
        format_feedback_prompt(
            base_prompt=_BASE,
            previous_kernel_code="",
            feedback=IncorrectFeedback(error_message="x"),
        )


def test_long_compile_error_is_head_truncated() -> None:
    long_err = "A" * 5_000
    feedback = CompileFailedFeedback(compilation_error=long_err)
    prompt = format_feedback_prompt(
        base_prompt=_BASE, previous_kernel_code=_PARENT, feedback=feedback
    )
    assert "[truncated]" in prompt
    # Head-truncation keeps the start of the error.
    assert prompt.count("A") < 5_000


def test_long_traceback_is_tail_truncated() -> None:
    # Build a traceback with a unique deep-frame marker.
    traceback_text = "frame_top\n" * 1_000 + "DEEP_FRAME_MARKER"
    feedback = RuntimeErrorFeedback(
        runtime_error_name="RuntimeError",
        runtime_error="boom",
        traceback=traceback_text,
    )
    prompt = format_feedback_prompt(
        base_prompt=_BASE, previous_kernel_code=_PARENT, feedback=feedback
    )
    # Tail-truncation preserves the deepest frame (most actionable).
    assert "DEEP_FRAME_MARKER" in prompt
    assert "[truncated]" in prompt
