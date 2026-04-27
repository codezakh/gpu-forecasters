"""Pydantic round-trip + failure adapter for ``gpu_mode_kernel.core``.

Mirrors ``tests/causal_conv1d/test_core.py`` and ``tests/trimul/test_core.py``,
but exercises the *generic* discriminated union with a concrete
``CaseSpeedup`` subclass declared inline.
"""

from __future__ import annotations

import pytest
from pydantic import ConfigDict, TypeAdapter

from arid_badger.gpu_mode_kernel.core import (
    CaseSpeedupBase,
    CompileFailedFeedback,
    IncorrectFeedback,
    KernelExecResult,
    KernelExecutionFeedback,
    RuntimeErrorFeedback,
    SuccessFeedback,
    failure_feedback_from_exec_result,
)


class FixtureCaseSpeedup(CaseSpeedupBase):
    """Minimal CaseSpeedup subclass used for round-trip testing."""

    model_config = ConfigDict(frozen=True)

    shape_a: int
    shape_b: int


_FixtureFeedback = KernelExecutionFeedback[FixtureCaseSpeedup]
_adapter: TypeAdapter[_FixtureFeedback] = TypeAdapter(_FixtureFeedback)


def _roundtrip(fb: _FixtureFeedback) -> _FixtureFeedback:
    return _adapter.validate_python(_adapter.dump_python(fb))


def test_success_roundtrip_carries_kernel_specific_shape_fields() -> None:
    fb = SuccessFeedback[FixtureCaseSpeedup](
        aggregated_speedup=1.8,
        aggregation_method="geomean",
        per_case_speedups=[
            FixtureCaseSpeedup(
                shape_a=128, shape_b=256,
                speedup=2.0, runtime_ns=1_000_000.0, ref_runtime_ns=2_000_000.0,
            ),
            FixtureCaseSpeedup(
                shape_a=512, shape_b=1024,
                speedup=1.6, runtime_ns=3_000_000.0, ref_runtime_ns=4_800_000.0,
            ),
        ],
    )
    rt = _roundtrip(fb)
    assert isinstance(rt, SuccessFeedback)
    assert rt == fb
    assert rt.per_case_speedups[0].shape_a == 128


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
    result = KernelExecResult(
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
    result = KernelExecResult(
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
    result = KernelExecResult(
        correct=False,
        runtime_ns=0.0,
        ref_runtime_ns=0.0,
        failure_kind="compile_failed",
        compilation_error="SyntaxError",
    )
    fb = failure_feedback_from_exec_result(result)
    assert isinstance(fb, CompileFailedFeedback)


def test_failure_adapter_raises_on_passing_result() -> None:
    result = KernelExecResult(
        correct=True,
        runtime_ns=500_000.0,
        ref_runtime_ns=1_000_000.0,
    )
    with pytest.raises(ValueError, match="failure_kind='none'"):
        failure_feedback_from_exec_result(result)
