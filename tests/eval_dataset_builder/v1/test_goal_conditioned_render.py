"""Pure-render tests for the goal-conditioned mutation prompt.

These exercise ``render_prompt`` directly — no asyncio loop, no
litellm, no Modal. They pin the prompt copy as a spec: any future
edit to ``mutation.jinja`` (or the band-neutral body substitutions)
that drops the goal framing or reintroduces max-speed language will
fail one of these tests.

The pack used as a fixture is ``TRIMUL_PACK`` because it ships with
the canonical body that these substitutions are validated against.
The test is pack-shaped, not pack-specific — the same prompt assembly
applies to any ``KernelPack``.
"""

from __future__ import annotations

from arid_badger.eval_dataset_builder.v1.goal_conditioned_mutation.provider import (
    render_prompt,
)
from arid_badger.gpu_mode_kernel.core import (
    CompileFailedFeedback,
    GpuModeKernelObservation,
    IncorrectFeedback,
    InfrastructureFailureFeedback,
    RuntimeErrorFeedback,
    SuccessFeedback,
)
from arid_badger.gpu_mode_kernel.packs.trimul import TRIMUL_PACK, TriMulCaseSpeedup
from arid_badger.gpu_mode_kernel.prompts import extract_last_python_codeblock
from arid_badger.hill_climbing.domain import Evaluation
from arid_badger.landscape_map.v1.domain import SpeedupBin


_GPU = "A100-SXM4-80GB"
_TRITON = "3.3.1"


def _eval(
    feedback: (
        SuccessFeedback[TriMulCaseSpeedup]
        | CompileFailedFeedback
        | RuntimeErrorFeedback
        | IncorrectFeedback
        | InfrastructureFailureFeedback
    ),
    reward: float | None,
) -> Evaluation[GpuModeKernelObservation[TriMulCaseSpeedup]]:
    obs = GpuModeKernelObservation[TriMulCaseSpeedup](feedback=feedback)
    return Evaluation[GpuModeKernelObservation[TriMulCaseSpeedup]](
        observation=obs, reward=reward
    )


def _case(speedup: float, seqlen: int = 256) -> TriMulCaseSpeedup:
    return TriMulCaseSpeedup(
        seqlen=seqlen,
        bs=1,
        dim=128,
        hiddendim=128,
        nomask=True,
        distribution="normal",
        speedup=speedup,
        runtime_ns=1000.0 / speedup,
        ref_runtime_ns=1000.0,
    )


_FORBIDDEN_PHRASES = [
    "highly optimized",
    "as fast as possible",
    "optimize runtime for these",
    "slowest first",
    # We deliberately replace the "iteratively optimizing runtime" framing
    # with "target band."
    "iteratively optimizing runtime",
]


def _assert_no_max_speed_language(prompt: str) -> None:
    for phrase in _FORBIDDEN_PHRASES:
        assert phrase not in prompt, (
            f"prompt contains forbidden max-speed phrase: {phrase!r}"
        )


def _assert_target_band_present(prompt: str, target_bin: SpeedupBin) -> None:
    assert target_bin.name in prompt or "target band" in prompt.lower(), (
        "prompt does not advertise the target bin"
    )
    assert "Target performance band" in prompt
    assert "midpoint" in prompt
    assert "Faster is *not* better" in prompt


def test_render_root_no_parent() -> None:
    prompt = render_prompt(
        pack=TRIMUL_PACK,
        target_bin=SpeedupBin.MINOR_SPEEDUP,
        parent_code=None,
        evaluation=None,
        gpu_name=_GPU,
        triton_version=_TRITON,
    )
    _assert_no_max_speed_language(prompt)
    _assert_target_band_present(prompt, SpeedupBin.MINOR_SPEEDUP)
    # SpeedupBin.from_speedup uses ``floor(2*log2(S)) + 4``, so bin 5
    # (MINOR_SPEEDUP) covers S ∈ [2^0.5, 2^1) = [1.41×, 2.00×).
    assert "1.41×" in prompt
    assert "2.00×" in prompt
    # No parent feedback section when there is no parent.
    assert "Here is your latest implementation" not in prompt
    assert "Compilation error" not in prompt
    assert "Per-case breakdown" not in prompt
    # Rules block is rendered with the gpu/triton vars.
    assert _GPU in prompt
    assert _TRITON in prompt


def test_render_success_above_band() -> None:
    feedback: SuccessFeedback[TriMulCaseSpeedup] = SuccessFeedback(
        aggregated_speedup=7.2,
        aggregation_method="geomean",
        per_case_speedups=[_case(7.5, seqlen=256), _case(6.8, seqlen=512)],
    )
    prompt = render_prompt(
        pack=TRIMUL_PACK,
        target_bin=SpeedupBin.MINOR_SPEEDUP,  # band [1.41, 2.00), parent at 7.2 is far above
        parent_code="def custom_kernel(data): pass",
        evaluation=_eval(feedback, reward=7.2),
        gpu_name=_GPU,
        triton_version=_TRITON,
    )
    _assert_no_max_speed_language(prompt)
    _assert_target_band_present(prompt, SpeedupBin.MINOR_SPEEDUP)
    assert "Here is your latest implementation" in prompt
    assert "def custom_kernel(data): pass" in prompt
    assert "Per-case breakdown (sorted by distance from the target midpoint)" in prompt
    # The success-arm trailing instruction should advise *slowing down*
    # because the parent is above the band.
    assert "above the band" in prompt
    assert "too fast" in prompt
    assert "slow it down" in prompt


def test_render_success_below_band() -> None:
    feedback: SuccessFeedback[TriMulCaseSpeedup] = SuccessFeedback(
        aggregated_speedup=0.5,
        aggregation_method="geomean",
        per_case_speedups=[_case(0.5, seqlen=256)],
    )
    prompt = render_prompt(
        pack=TRIMUL_PACK,
        target_bin=SpeedupBin.MINOR_SPEEDUP,  # parent at 0.5 is below
        parent_code="def custom_kernel(data): pass",
        evaluation=_eval(feedback, reward=0.5),
        gpu_name=_GPU,
        triton_version=_TRITON,
    )
    _assert_no_max_speed_language(prompt)
    assert "below the band" in prompt
    assert "too slow" in prompt


def test_render_compile_failed() -> None:
    feedback = CompileFailedFeedback(compilation_error="SyntaxError: invalid syntax")
    prompt = render_prompt(
        pack=TRIMUL_PACK,
        target_bin=SpeedupBin.HIGH_SPEEDUP,
        parent_code="broken code",
        evaluation=_eval(feedback, reward=None),
        gpu_name=_GPU,
        triton_version=_TRITON,
    )
    _assert_no_max_speed_language(prompt)
    _assert_target_band_present(prompt, SpeedupBin.HIGH_SPEEDUP)
    assert "Compilation error:" in prompt
    assert "SyntaxError: invalid syntax" in prompt
    # Bin 7 (HIGH_SPEEDUP) = [2^1.5, 2^2) = [2.83×, 4.00×)
    assert "2.83×" in prompt
    assert "4.00×" in prompt


def test_render_runtime_error() -> None:
    feedback = RuntimeErrorFeedback(
        runtime_error_name="RuntimeError",
        runtime_error="CUDA out of memory",
        traceback='Traceback (most recent call last):\n  File "x.py", line 1, in <module>',
    )
    prompt = render_prompt(
        pack=TRIMUL_PACK,
        target_bin=SpeedupBin.MINOR_SLOWDOWN,
        parent_code="def custom_kernel(data): raise RuntimeError",
        evaluation=_eval(feedback, reward=None),
        gpu_name=_GPU,
        triton_version=_TRITON,
    )
    _assert_no_max_speed_language(prompt)
    _assert_target_band_present(prompt, SpeedupBin.MINOR_SLOWDOWN)
    assert "Error type: RuntimeError" in prompt
    assert "CUDA out of memory" in prompt
    assert "Traceback" in prompt


def test_render_incorrect() -> None:
    feedback = IncorrectFeedback(error_message="output mismatch: max abs diff 0.5")
    prompt = render_prompt(
        pack=TRIMUL_PACK,
        target_bin=SpeedupBin.SIGNIFICANT_SLOWDOWN,
        parent_code="def custom_kernel(data): return wrong",
        evaluation=_eval(feedback, reward=None),
        gpu_name=_GPU,
        triton_version=_TRITON,
    )
    _assert_no_max_speed_language(prompt)
    _assert_target_band_present(prompt, SpeedupBin.SIGNIFICANT_SLOWDOWN)
    assert "Correctness issue:" in prompt
    assert "output mismatch: max abs diff 0.5" in prompt


def test_render_infrastructure_failure_drops_parent() -> None:
    feedback = InfrastructureFailureFeedback(reason="modal timeout")
    prompt = render_prompt(
        pack=TRIMUL_PACK,
        target_bin=SpeedupBin.HIGH_SPEEDUP,
        parent_code="def custom_kernel(data): pass",
        evaluation=_eval(feedback, reward=None),
        gpu_name=_GPU,
        triton_version=_TRITON,
    )
    _assert_no_max_speed_language(prompt)
    _assert_target_band_present(prompt, SpeedupBin.HIGH_SPEEDUP)
    # Infra failures carry no signal — the prompt should not show the
    # parent's code.
    assert "Here is your latest implementation" not in prompt


def test_band_neutral_substitutions_applied() -> None:
    """Confirm the two phrase substitutions on ``base_prompt`` actually
    fire on TriMul's body. Both source phrases are present in
    ``TRIMUL_PACK.kernel_description_body`` and must be replaced before
    rendering.
    """
    prompt = render_prompt(
        pack=TRIMUL_PACK,
        target_bin=SpeedupBin.MINOR_SPEEDUP,
        parent_code=None,
        evaluation=None,
        gpu_name=_GPU,
        triton_version=_TRITON,
    )
    assert "highly optimized" not in prompt
    assert "optimize runtime for these" not in prompt
    assert "Triton engineer translating PyTorch code into Triton kernel code that hits a target performance band" in prompt
    assert "Test cases (your kernel will be measured on these for both correctness and runtime):" in prompt


def test_extract_last_python_codeblock_round_trip() -> None:
    """Sanity check: the shared library helper still works on the kind
    of LLM response the provider expects (reasoning + final block)."""
    response = (
        "Here is my reasoning... ```python\n# draft\npass\n``` "
        "and the final answer:\n\n```python\ndef custom_kernel(data):\n    return data\n```"
    )
    code = extract_last_python_codeblock(response)
    assert code is not None
    assert "def custom_kernel(data):" in code
    assert "draft" not in code
