"""Pydantic round-trip for the causal conv1d feedback discriminated union.

Mirrors ``tests/trimul/test_core.py``. Tests the discriminator wiring +
``failure_feedback_from_exec_result`` adapter — actual domain logic
that we wrote, not framework behavior.
"""

from __future__ import annotations

import pytest
from pydantic import TypeAdapter

from gpu_forecasters.causal_conv1d.core import (
    CaseSpeedup,
    CausalConv1dExecResult,
    CausalConv1dKernelExecutionFeedback,
    CompileFailedFeedback,
    IncorrectFeedback,
    RuntimeErrorFeedback,
    SuccessFeedback,
    failure_feedback_from_exec_result,
)


_adapter: TypeAdapter[CausalConv1dKernelExecutionFeedback] = TypeAdapter(
    CausalConv1dKernelExecutionFeedback
)


def _roundtrip(
    fb: CausalConv1dKernelExecutionFeedback,
) -> CausalConv1dKernelExecutionFeedback:
    return _adapter.validate_python(_adapter.dump_python(fb))


def test_success_roundtrip() -> None:
    fb = SuccessFeedback(
        aggregated_speedup=1.8,
        aggregation_method="geomean",
        per_case_speedups=[
            CaseSpeedup(
                B=1, D=1536, S=2048, W=4,
                speedup=2.0, runtime_ns=1_000_000.0, ref_runtime_ns=2_000_000.0,
            ),
            CaseSpeedup(
                B=1, D=2560, S=4096, W=4,
                speedup=1.6, runtime_ns=3_000_000.0, ref_runtime_ns=4_800_000.0,
            ),
        ],
    )
    rt = _roundtrip(fb)
    assert isinstance(rt, SuccessFeedback)
    assert rt == fb


def test_incorrect_roundtrip() -> None:
    fb = IncorrectFeedback(error_message="diff too big")
    rt = _roundtrip(fb)
    assert isinstance(rt, IncorrectFeedback)
    assert rt.error_message == "diff too big"


def test_runtime_error_roundtrip() -> None:
    fb = RuntimeErrorFeedback(
        runtime_error_name="RuntimeError",
        runtime_error="boom",
        traceback="Traceback (...)",
    )
    rt = _roundtrip(fb)
    assert isinstance(rt, RuntimeErrorFeedback)
    assert rt.runtime_error == "boom"


def test_compile_failed_roundtrip() -> None:
    fb = CompileFailedFeedback(compilation_error="SyntaxError: invalid syntax")
    rt = _roundtrip(fb)
    assert isinstance(rt, CompileFailedFeedback)
    assert "SyntaxError" in rt.compilation_error


def test_failure_adapter_incorrect() -> None:
    result = CausalConv1dExecResult(
        correct=False,
        runtime_ns=0.0,
        ref_runtime_ns=0.0,
        failure_kind="incorrect",
        error_message="mismatch",
    )
    fb = failure_feedback_from_exec_result(result)
    assert isinstance(fb, IncorrectFeedback)
    assert fb.error_message == "mismatch"


def test_failure_adapter_runtime_error() -> None:
    result = CausalConv1dExecResult(
        correct=False,
        runtime_ns=0.0,
        ref_runtime_ns=0.0,
        failure_kind="runtime_error",
        runtime_error_name="RuntimeError",
        runtime_error="boom",
        traceback="tb",
    )
    fb = failure_feedback_from_exec_result(result)
    assert isinstance(fb, RuntimeErrorFeedback)
    assert fb.runtime_error == "boom"


def test_failure_adapter_compile_failed() -> None:
    result = CausalConv1dExecResult(
        correct=False,
        runtime_ns=0.0,
        ref_runtime_ns=0.0,
        failure_kind="compile_failed",
        compilation_error="SyntaxError",
    )
    fb = failure_feedback_from_exec_result(result)
    assert isinstance(fb, CompileFailedFeedback)


def test_failure_adapter_raises_on_passing_result() -> None:
    result = CausalConv1dExecResult(
        correct=True,
        runtime_ns=500_000.0,
        ref_runtime_ns=1_000_000.0,
    )
    with pytest.raises(ValueError, match="failure_kind='none'"):
        failure_feedback_from_exec_result(result)
