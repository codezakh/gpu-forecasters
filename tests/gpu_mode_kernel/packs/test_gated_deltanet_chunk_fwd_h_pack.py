"""Snapshot tests for the GDN chunk-fwd-h pack.

There is no legacy formatter to byte-compare against (this is the
first pack written natively for the gpu_mode_kernel abstraction with
no prior per-kernel mirror), so these tests pin the prompt-line
format and the description body shape directly. They will fail if a
casual edit perturbs the LLM-facing wire format — same defensive
intent as the TriMul cross-validation, just without a second source
of truth.

Tested logic:
- ``GdnChunkFwdHCaseSpeedup.from_exec_result`` populates each shape
  field from the test args dict and computes ``speedup = ref/cand``.
- ``GdnChunkFwdHCaseSpeedup.format_for_prompt`` emits the exact line
  shape the LLM will see in the success-arm feedback.
- ``GDN_CHUNK_FWD_H_PACK.kernel_description_body`` ends with the
  auto-generated test/benchmark cases listing.
"""

from __future__ import annotations

from arid_badger.gpu_mode_kernel.core import KernelExecResult
from arid_badger.gpu_mode_kernel.packs.gated_deltanet_chunk_fwd_h import (
    BENCHMARK_CASES,
    CORRECTNESS_CASES,
    GDN_CHUNK_FWD_H_PACK,
    GdnChunkFwdHCaseSpeedup,
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

    cs = GdnChunkFwdHCaseSpeedup.from_exec_result(test_args, exec_result)

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

    cs = GdnChunkFwdHCaseSpeedup.from_exec_result(test_args, exec_result)

    assert cs.speedup == 0.0


def test_case_speedup_format_for_prompt_emits_expected_line_shape() -> None:
    cs = GdnChunkFwdHCaseSpeedup(
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

    # Pin the exact format the LLM sees. Drift here changes the LLM's
    # input distribution silently.
    assert line == "B=2, T=1024, H=3, K=64, V=64: 2.500x (ref: 10.0μs, candidate: 4.0μs)"


def test_kernel_description_body_lists_all_cases() -> None:
    body = GDN_CHUNK_FWD_H_PACK.kernel_description_body

    # Auto-generated case lines. Each correctness/benchmark case must
    # appear with the exact (B, T, H, K, V) tuple it'll be scored on.
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
    assert GDN_CHUNK_FWD_H_PACK.name == "gdn_chunk_fwd_h"
    assert GDN_CHUNK_FWD_H_PACK.modal_app_name == "arid-badger-gdn-chunk-fwd-h"
    assert GDN_CHUNK_FWD_H_PACK.determinism_ctx is None
    assert GDN_CHUNK_FWD_H_PACK.case_speedup_type is GdnChunkFwdHCaseSpeedup
