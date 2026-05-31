"""Cross-validation: ``TRIMUL_PACK`` prompt formatting vs legacy.

The new ``format_feedback_prompt`` (kernel-agnostic) plus
``TriMulCaseSpeedup.format_for_prompt`` (pack-specific) must produce
byte-equivalent output to the legacy
``format_trimul_feedback_mutation_prompt``. Both code paths feed the
same LLM with the same prompt-engineering baseline; any silent drift
in the per-case formatter (e.g. ``dist=`` vs ``distribution=``)
shifts the LLM's distribution and confounds A/B comparisons against
the legacy stack.

These are pure unit tests — no Modal, no LLM. They fail loudly if
the formatters disagree on a single character.
"""

from __future__ import annotations

from gpu_forecasters.gpu_mode_kernel.core import (
    CompileFailedFeedback,
    IncorrectFeedback,
    RuntimeErrorFeedback,
    SuccessFeedback,
)
from gpu_forecasters.gpu_mode_kernel.packs.trimul import TriMulCaseSpeedup
from gpu_forecasters.gpu_mode_kernel.prompts import format_feedback_prompt
from gpu_forecasters.hill_climbing.mutation_providers.trimul_feedback_mutation import (
    format_trimul_feedback_mutation_prompt,
)
from gpu_forecasters.trimul.core import (
    CaseSpeedup as LegacyCaseSpeedup,
)
from gpu_forecasters.trimul.core import (
    CompileFailedFeedback as LegacyCompileFailedFeedback,
)
from gpu_forecasters.trimul.core import (
    IncorrectFeedback as LegacyIncorrectFeedback,
)
from gpu_forecasters.trimul.core import (
    RuntimeErrorFeedback as LegacyRuntimeErrorFeedback,
)
from gpu_forecasters.trimul.core import (
    SuccessFeedback as LegacySuccessFeedback,
)


_BASE = "BASE_PROMPT_BODY"
_PARENT = "def custom_kernel(data):\n    return data"


def test_compile_failed_matches_legacy() -> None:
    new_fb = CompileFailedFeedback(
        compilation_error="SyntaxError: invalid syntax (line 3)"
    )
    legacy_fb = LegacyCompileFailedFeedback(
        compilation_error="SyntaxError: invalid syntax (line 3)"
    )
    assert format_feedback_prompt(
        base_prompt=_BASE, previous_kernel_code=_PARENT, feedback=new_fb
    ) == format_trimul_feedback_mutation_prompt(
        base_prompt=_BASE, previous_kernel_code=_PARENT, feedback=legacy_fb
    )


def test_runtime_error_matches_legacy() -> None:
    new_fb = RuntimeErrorFeedback(
        runtime_error_name="RuntimeError",
        runtime_error="boom",
        traceback="frame_top\nframe_bottom",
    )
    legacy_fb = LegacyRuntimeErrorFeedback(
        runtime_error_name="RuntimeError",
        runtime_error="boom",
        traceback="frame_top\nframe_bottom",
    )
    assert format_feedback_prompt(
        base_prompt=_BASE, previous_kernel_code=_PARENT, feedback=new_fb
    ) == format_trimul_feedback_mutation_prompt(
        base_prompt=_BASE, previous_kernel_code=_PARENT, feedback=legacy_fb
    )


def test_incorrect_matches_legacy() -> None:
    new_fb = IncorrectFeedback(error_message="elements differ at idx 5")
    legacy_fb = LegacyIncorrectFeedback(error_message="elements differ at idx 5")
    assert format_feedback_prompt(
        base_prompt=_BASE, previous_kernel_code=_PARENT, feedback=new_fb
    ) == format_trimul_feedback_mutation_prompt(
        base_prompt=_BASE, previous_kernel_code=_PARENT, feedback=legacy_fb
    )


def test_success_arm_matches_legacy_per_case_format() -> None:
    """The success arm is the highest-risk one because each pack has
    its own per-case formatter — ``dist=`` vs ``distribution=`` is the
    bug class this is guarding against.
    """
    cases_args = [
        # (seqlen, bs, dim, hiddendim, nomask, distribution, speedup, runtime_ns, ref_runtime_ns)
        (256, 2, 128, 128, True, "normal", 4.000, 1_000.0, 4_000.0),
        (768, 1, 128, 128, True, "cauchy", 1.500, 2_000.0, 3_000.0),
        (1024, 1, 384, 128, False, "normal", 2.000, 1_500.0, 3_000.0),
    ]
    new_cases = [
        TriMulCaseSpeedup(
            seqlen=s, bs=b, dim=d, hiddendim=h, nomask=n, distribution=dist,
            speedup=sp, runtime_ns=r, ref_runtime_ns=rr,
        )
        for s, b, d, h, n, dist, sp, r, rr in cases_args
    ]
    legacy_cases = [
        LegacyCaseSpeedup(
            seqlen=s, bs=b, dim=d, hiddendim=h, nomask=n, distribution=dist,
            speedup=sp, runtime_ns=r, ref_runtime_ns=rr,
        )
        for s, b, d, h, n, dist, sp, r, rr in cases_args
    ]
    new_fb = SuccessFeedback[TriMulCaseSpeedup](
        aggregated_speedup=2.289,
        aggregation_method="geomean",
        per_case_speedups=new_cases,
    )
    legacy_fb = LegacySuccessFeedback(
        aggregated_speedup=2.289,
        aggregation_method="geomean",
        per_case_speedups=legacy_cases,
    )
    assert format_feedback_prompt(
        base_prompt=_BASE, previous_kernel_code=_PARENT, feedback=new_fb
    ) == format_trimul_feedback_mutation_prompt(
        base_prompt=_BASE, previous_kernel_code=_PARENT, feedback=legacy_fb
    )
