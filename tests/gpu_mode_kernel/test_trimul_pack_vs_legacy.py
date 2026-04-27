"""Regression check: ``TRIMUL_PACK`` via gpu_mode_kernel vs the legacy
``arid_badger.trimul.scoring.score`` path.

This is the Path-B correctness gate. The new abstraction must produce
the same scoring output as the legacy per-kernel package on the same
candidate + same test case + same container.

To eliminate cross-container variance, both paths run back-to-back
inside one custom Modal cls (rather than going through the two
production benchmarker classes, which would land on different
containers). Three scenarios:

1. Syntactically-broken candidate → both paths produce
   ``CompileFailedFeedback`` with matching ``compilation_error``
   prefixes (the trailing module path differs because each tmpdir is
   unique, but the error type and source location should match).
2. Correct-but-trivially-wrong candidate (returns zeros) → both paths
   produce ``IncorrectFeedback``.
3. Legacy seed kernel (PyTorch TriMul reference) → both paths produce
   ``SuccessFeedback``-shaped results with timings agreeing to within
   25%. Some variance is unavoidable: the adaptive timing loop is
   noisy and the two calls don't run literally in parallel, but
   anything beyond ~25% on a stable kernel like TriMul indicates a
   real divergence in the abstractions.

Marked ``integration`` — opt-in via ``-m integration``.
"""

from __future__ import annotations

from typing import Any

import modal
import pytest

from arid_badger.gpu_mode_kernel.packs.trimul import (
    BENCHMARK_CASES as PACK_BENCHMARK_CASES,
)
from arid_badger.kernelbench.modal_image import image


# Dedicated app — distinct namespace from both ``arid-badger-trimul``
# (legacy) and ``arid-badger-trimul-v2pack`` (new pack) so this test
# doesn't fight either production app for container leases.
_app = modal.App("arid-badger-trimul-regression-check")


_BROKEN_CANDIDATE = """\
this is not valid python !!!
"""


_INCORRECT_CANDIDATE = """\
import torch
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    # Returns zeros — definitely wrong on every case.
    input_tensor, mask, weights, config = data
    return torch.zeros_like(input_tensor)
"""


@_app.cls(
    image=image,
    gpu="A100-80GB",
    timeout=1200,
    max_containers=1,
    single_use_containers=True,
    scaledown_window=2,
)
class _RegressionRunner:
    """Runs both scoring paths on the same container against the same
    candidate + test case. Returns the legacy ``TriMulExecResult`` and
    the new ``KernelExecResult`` as plain dicts so they cross the wire
    cleanly without import-side-effect surprises.
    """

    @modal.method()
    def run_both(
        self,
        candidate_code: str,
        test_case: dict[str, object],
    ) -> dict[str, Any]:
        from typing import cast

        from arid_badger.gpu_mode_kernel.packs.trimul import TRIMUL_PACK
        from arid_badger.gpu_mode_kernel.scoring import (
            score_one_case as new_score_one_case,
        )
        from arid_badger.trimul.cases import TriMulTestArgs as LegacyTestArgs
        from arid_badger.trimul.scoring import score as legacy_score
        from arid_badger.typing_utils import is_ok

        # Run legacy first.
        legacy_outcome = legacy_score(
            candidate_code,
            cast(LegacyTestArgs, cast(object, test_case)),
            max_repeats=20,
            max_time_ns=2e9,
        )
        # Then new pack.
        new_outcome = new_score_one_case(
            pack=TRIMUL_PACK,
            mutated_kernel_code=candidate_code,
            test_args=cast(Any, test_case),
            max_repeats=20,
            max_time_ns=2e9,
        )

        legacy = (
            legacy_outcome.unwrap().model_dump()
            if is_ok(legacy_outcome)
            else {"_err": str(legacy_outcome.unwrap_err())}
        )
        new = (
            new_outcome.unwrap().model_dump()
            if is_ok(new_outcome)
            else {"_err": str(new_outcome.unwrap_err())}
        )
        return {"legacy": legacy, "new": new}


@pytest.mark.integration
def test_compile_failed_arm_matches() -> None:
    """Syntactically-broken candidate → both paths report compile_failed."""
    case = PACK_BENCHMARK_CASES[0]
    with _app.run():
        runner = _RegressionRunner()
        result = runner.run_both.remote(
            candidate_code=_BROKEN_CANDIDATE,
            test_case=dict(case),  # ty: ignore[invalid-argument-type]
        )

    legacy = result["legacy"]
    new = result["new"]

    assert "_err" not in legacy and "_err" not in new
    assert legacy["correct"] is False
    assert new["correct"] is False
    assert legacy["failure_kind"] == "compile_failed"
    assert new["failure_kind"] == "compile_failed"
    # The two paths construct compile-failed messages identically:
    # ``f"Candidate source has a syntax error: {exc}"``. Asserting
    # exact equality is the strongest regression check available
    # since the exception text is deterministic (Python's tokenizer
    # output for a fixed broken source) and tmpdir prefix is the
    # same for both (``trimul-candidate-`` derived from pack.name).
    assert legacy["compilation_error"] == new["compilation_error"]
    assert "syntax error" in legacy["compilation_error"]


@pytest.mark.integration
def test_incorrect_arm_matches() -> None:
    """Trivially-wrong candidate (zeros) → both paths report incorrect."""
    case = PACK_BENCHMARK_CASES[0]
    with _app.run():
        runner = _RegressionRunner()
        result = runner.run_both.remote(
            candidate_code=_INCORRECT_CANDIDATE,
            test_case=dict(case),  # ty: ignore[invalid-argument-type]
        )

    legacy = result["legacy"]
    new = result["new"]

    assert "_err" not in legacy and "_err" not in new
    assert legacy["correct"] is False
    assert new["correct"] is False
    assert legacy["failure_kind"] == "incorrect"
    assert new["failure_kind"] == "incorrect"


@pytest.mark.integration
def test_success_arm_timings_agree_within_tolerance() -> None:
    """Legacy seed kernel through both paths → SuccessFeedback shape,
    candidate and reference timings agree to within 25%.

    The bound is loose because the adaptive timing loop is stochastic
    and the two scoring calls don't run literally simultaneously.
    A regression in the abstraction would surface as a much larger
    divergence (different code path / wrong reference / etc.).
    """
    from arid_badger.gpu_mode_kernel.packs.trimul import TRIMUL_PACK

    case = PACK_BENCHMARK_CASES[0]
    with _app.run():
        runner = _RegressionRunner()
        result = runner.run_both.remote(
            candidate_code=TRIMUL_PACK.seed_kernel_code,
            test_case=dict(case),  # ty: ignore[invalid-argument-type]
        )

    legacy = result["legacy"]
    new = result["new"]

    assert "_err" not in legacy and "_err" not in new
    assert legacy["correct"] is True, f"legacy seed not correct: {legacy}"
    assert new["correct"] is True, f"new seed not correct: {new}"

    legacy_cand = legacy["runtime_ns"]
    new_cand = new["runtime_ns"]
    legacy_ref = legacy["ref_runtime_ns"]
    new_ref = new["ref_runtime_ns"]

    def _within(a: float, b: float, tol: float) -> bool:
        if min(a, b) <= 0:
            return False
        return abs(a - b) / min(a, b) <= tol

    assert _within(legacy_cand, new_cand, 0.25), (
        f"candidate timing divergence: legacy={legacy_cand:.0f}ns "
        f"new={new_cand:.0f}ns (diff "
        f"{abs(legacy_cand - new_cand) / min(legacy_cand, new_cand) * 100:.1f}%)"
    )
    assert _within(legacy_ref, new_ref, 0.25), (
        f"reference timing divergence: legacy={legacy_ref:.0f}ns "
        f"new={new_ref:.0f}ns (diff "
        f"{abs(legacy_ref - new_ref) / min(legacy_ref, new_ref) * 100:.1f}%)"
    )
