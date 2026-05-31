"""Snapshot tests for the GDN recompute-w-u pack.

Same defensive intent as the GDN h/o snapshot tests — there is no
legacy formatter to byte-compare against, so these tests pin the
prompt-line format and the description body shape directly. They will
fail if a casual edit perturbs the LLM-facing wire format.

Tested logic:
- ``GdnRecomputeWUCaseSpeedup.from_exec_result`` populates each shape
  field from the test args dict and computes ``speedup = ref/cand``.
- ``GdnRecomputeWUCaseSpeedup.format_for_prompt`` emits the exact line
  shape the LLM will see in the success-arm feedback.
- ``GDN_RECOMPUTE_W_U_PACK.kernel_description_body`` ends with the
  auto-generated test/benchmark cases listing.
"""

from __future__ import annotations

from gpu_forecasters.gpu_mode_kernel.core import KernelExecResult
from gpu_forecasters.gpu_mode_kernel.packs.gated_deltanet_recompute_w_u import (
    BENCHMARK_CASES,
    CORRECTNESS_CASES,
    GDN_RECOMPUTE_W_U_PACK,
    GdnRecomputeWUCaseSpeedup,
)


def _exec_result(*, runtime_ns: float, ref_runtime_ns: float) -> KernelExecResult:
    return KernelExecResult(
        correct=True,
        runtime_ns=runtime_ns,
        ref_runtime_ns=ref_runtime_ns,
    )


def test_case_speedup_from_exec_result_populates_shape_and_speedup() -> None:
    test_args = {"B": 2, "T": 1024, "H": 3, "K": 64, "V": 64, "seed": 2146}
    exec_result = _exec_result(runtime_ns=2_000.0, ref_runtime_ns=10_000.0)

    cs = GdnRecomputeWUCaseSpeedup.from_exec_result(test_args, exec_result)

    assert cs.B == 2
    assert cs.T == 1024
    assert cs.H == 3
    assert cs.K == 64
    assert cs.V == 64
    assert cs.runtime_ns == 2_000.0
    assert cs.ref_runtime_ns == 10_000.0
    assert cs.speedup == 5.0


def test_case_speedup_from_exec_result_zero_runtime_yields_zero_speedup() -> None:
    """Guard against div-by-zero — matches the convention in the other packs."""
    test_args = {"B": 1, "T": 64, "H": 1, "K": 64, "V": 64, "seed": 0}
    exec_result = _exec_result(runtime_ns=0.0, ref_runtime_ns=10_000.0)

    cs = GdnRecomputeWUCaseSpeedup.from_exec_result(test_args, exec_result)

    assert cs.speedup == 0.0


def test_case_speedup_format_for_prompt_emits_expected_line_shape() -> None:
    cs = GdnRecomputeWUCaseSpeedup(
        B=2,
        T=1024,
        H=3,
        K=64,
        V=64,
        speedup=2.5,
        runtime_ns=4_000.0,
        ref_runtime_ns=10_000.0,
    )

    line = cs.format_for_prompt()

    assert line == "B=2, T=1024, H=3, K=64, V=64: 2.500x (ref: 10.0μs, candidate: 4.0μs)"


def test_kernel_description_body_lists_all_cases() -> None:
    body = GDN_RECOMPUTE_W_U_PACK.kernel_description_body

    for c in CORRECTNESS_CASES + BENCHMARK_CASES:
        expected_line = (
            f'  - {{"B": {c["B"]}, "T": {c["T"]}, "H": {c["H"]}, '
            f'"K": {c["K"]}, "V": {c["V"]}}}'
        )
        assert expected_line in body, f"missing case line: {expected_line}"

    assert "Benchmark cases for runtime" in body
    assert "Test cases for correctness:" in body


def test_pack_metadata() -> None:
    """Pin the pack's identity fields — Modal app name in particular
    is a registry namespace; renaming silently risks colliding with a
    different pack's container set."""
    assert GDN_RECOMPUTE_W_U_PACK.name == "gdn_recompute_w_u"
    assert GDN_RECOMPUTE_W_U_PACK.modal_app_name == "arid-badger-gdn-recompute-w-u"
    assert GDN_RECOMPUTE_W_U_PACK.determinism_ctx is None
    assert GDN_RECOMPUTE_W_U_PACK.case_speedup_type is GdnRecomputeWUCaseSpeedup
