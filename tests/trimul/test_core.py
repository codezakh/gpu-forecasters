"""Pydantic round-trip for the TriMul feedback discriminated union.

Tests the discriminator wiring — actual domain logic that we wrote,
not framework behavior (per the ``tests must test real logic`` rule).
"""

from __future__ import annotations

from pydantic import TypeAdapter

from arid_badger.trimul.core import (
    CompileFailedFeedback,
    IncorrectFeedback,
    RuntimeErrorFeedback,
    SuccessFeedback,
    TriMulExecResult,
    TriMulKernelExecutionFeedback,
    execution_feedback_from_exec_result,
)


_adapter: TypeAdapter[TriMulKernelExecutionFeedback] = TypeAdapter(
    TriMulKernelExecutionFeedback
)


def _roundtrip(fb: TriMulKernelExecutionFeedback) -> TriMulKernelExecutionFeedback:
    return _adapter.validate_python(_adapter.dump_python(fb))


def test_success_roundtrip() -> None:
    fb = SuccessFeedback(runtime_ns=1_000_000.0, ref_runtime_ns=2_000_000.0, speedup=2.0)
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


def test_adapter_success() -> None:
    result = TriMulExecResult(
        correct=True, runtime_ns=500_000.0, ref_runtime_ns=1_000_000.0
    )
    fb = execution_feedback_from_exec_result(exec_result=result, speedup=2.0)
    assert isinstance(fb, SuccessFeedback)
    assert fb.speedup == 2.0


def test_adapter_incorrect() -> None:
    result = TriMulExecResult(
        correct=False,
        runtime_ns=0.0,
        ref_runtime_ns=0.0,
        failure_kind="incorrect",
        error_message="mismatch",
    )
    fb = execution_feedback_from_exec_result(exec_result=result, speedup=0.0)
    assert isinstance(fb, IncorrectFeedback)
    assert fb.error_message == "mismatch"


def test_adapter_runtime_error() -> None:
    result = TriMulExecResult(
        correct=False,
        runtime_ns=0.0,
        ref_runtime_ns=0.0,
        failure_kind="runtime_error",
        runtime_error_name="RuntimeError",
        runtime_error="boom",
        traceback="tb",
    )
    fb = execution_feedback_from_exec_result(exec_result=result, speedup=0.0)
    assert isinstance(fb, RuntimeErrorFeedback)
    assert fb.runtime_error == "boom"


def test_adapter_compile_failed() -> None:
    result = TriMulExecResult(
        correct=False,
        runtime_ns=0.0,
        ref_runtime_ns=0.0,
        failure_kind="compile_failed",
        compilation_error="SyntaxError",
    )
    fb = execution_feedback_from_exec_result(exec_result=result, speedup=0.0)
    assert isinstance(fb, CompileFailedFeedback)
