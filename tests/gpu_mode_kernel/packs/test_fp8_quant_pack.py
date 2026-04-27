"""Snapshot tests for the FP8 per-token-group quantization pack.

Same defensive intent as the GDN snapshot tests — pin the prompt-line
format and the description body shape so a casual edit can't perturb
the LLM-facing wire format silently.
"""

from __future__ import annotations

from arid_badger.gpu_mode_kernel.core import KernelExecResult
from arid_badger.gpu_mode_kernel.packs.fp8_quant import (
    BENCHMARK_CASES,
    CORRECTNESS_CASES,
    FP8_QUANT_PACK,
    Fp8QuantCaseSpeedup,
)


def _exec_result(*, runtime_ns: float, ref_runtime_ns: float) -> KernelExecResult:
    return KernelExecResult(
        correct=True,
        runtime_ns=runtime_ns,
        ref_runtime_ns=ref_runtime_ns,
    )


def test_case_speedup_from_exec_result_populates_shape_and_speedup() -> None:
    test_args = {
        "num_tokens": 256,
        "hidden_dim": 4096,
        "group_size": 128,
        "seed": 2146,
    }
    exec_result = _exec_result(runtime_ns=2_000.0, ref_runtime_ns=10_000.0)

    cs = Fp8QuantCaseSpeedup.from_exec_result(test_args, exec_result)

    assert cs.num_tokens == 256
    assert cs.hidden_dim == 4096
    assert cs.group_size == 128
    assert cs.runtime_ns == 2_000.0
    assert cs.ref_runtime_ns == 10_000.0
    assert cs.speedup == 5.0


def test_case_speedup_from_exec_result_zero_runtime_yields_zero_speedup() -> None:
    test_args = {
        "num_tokens": 1,
        "hidden_dim": 256,
        "group_size": 64,
        "seed": 0,
    }
    exec_result = _exec_result(runtime_ns=0.0, ref_runtime_ns=10_000.0)

    cs = Fp8QuantCaseSpeedup.from_exec_result(test_args, exec_result)

    assert cs.speedup == 0.0


def test_case_speedup_format_for_prompt_emits_expected_line_shape() -> None:
    cs = Fp8QuantCaseSpeedup(
        num_tokens=256,
        hidden_dim=4096,
        group_size=128,
        speedup=2.5,
        runtime_ns=4_000.0,
        ref_runtime_ns=10_000.0,
    )

    line = cs.format_for_prompt()

    assert line == (
        "num_tokens=256, hidden_dim=4096, group_size=128: "
        "2.500x (ref: 10.0μs, candidate: 4.0μs)"
    )


def test_kernel_description_body_lists_all_cases() -> None:
    body = FP8_QUANT_PACK.kernel_description_body

    for c in CORRECTNESS_CASES + BENCHMARK_CASES:
        expected_line = (
            f'  - {{"num_tokens": {c["num_tokens"]}, '
            f'"hidden_dim": {c["hidden_dim"]}, '
            f'"group_size": {c["group_size"]}}}'
        )
        assert expected_line in body, f"missing case line: {expected_line}"

    assert "Benchmark cases for runtime" in body
    assert "Test cases for correctness:" in body


def test_pack_metadata() -> None:
    """Pin the pack's identity fields — Modal app name in particular
    is a registry namespace; renaming silently risks colliding with a
    different pack's container set."""
    assert FP8_QUANT_PACK.name == "fp8_quant"
    assert FP8_QUANT_PACK.modal_app_name == "arid-badger-fp8-quant"
    assert FP8_QUANT_PACK.determinism_ctx is None
    assert FP8_QUANT_PACK.case_speedup_type is Fp8QuantCaseSpeedup
