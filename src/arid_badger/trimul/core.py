"""TriMul-specific execution feedback types and exec result.

Parallel to ``arid_badger.kernelbench.core`` but with TriMul-appropriate
fields: runtime in nanoseconds (matching the cuda.Event adaptive loop's
native unit) rather than microseconds, no KernelBench-specific fields
like compilation arch. A discriminated union over ``kind`` lets callers
pattern-match on outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Adaptive-loop stats (copied from ttt-discover eval.py lines 94-102).
# ---------------------------------------------------------------------------


@dataclass
class Stats:
    """Timing statistics from an adaptive benchmarking loop."""

    runs: int
    mean: float
    std: float
    err: float
    best: float
    worst: float


# ---------------------------------------------------------------------------
# Feedback union
# ---------------------------------------------------------------------------


class CompileFailedFeedback(BaseModel):
    """Module import / syntax error — the TriMul analogue of a compile failure.

    TriMul candidates are plain Python modules exposing ``custom_kernel``; a
    SyntaxError or NameError during module import stands in for nvcc's
    compile failure in KernelBench land.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["compile_failed"] = "compile_failed"
    compilation_error: str


class RuntimeErrorFeedback(BaseModel):
    """Candidate's ``custom_kernel`` raised at runtime."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["runtime_error"] = "runtime_error"
    runtime_error_name: str
    runtime_error: str
    traceback: str


class IncorrectFeedback(BaseModel):
    """Candidate ran but produced output outside tolerance of reference."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["incorrect"] = "incorrect"
    error_message: str


class SuccessFeedback(BaseModel):
    """Candidate ran and passed correctness; timings in nanoseconds."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["success"] = "success"
    runtime_ns: float
    ref_runtime_ns: float
    speedup: float


class InfrastructureFailureFeedback(BaseModel):
    """Scoring harness crashed before producing a verdict."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["infrastructure_failure"] = "infrastructure_failure"
    reason: str


TriMulKernelExecutionFeedback = Annotated[
    Union[
        CompileFailedFeedback,
        RuntimeErrorFeedback,
        IncorrectFeedback,
        SuccessFeedback,
    ],
    Field(discriminator="kind"),
]


# ---------------------------------------------------------------------------
# Exec result — the wire-format return type of ``score()``
# ---------------------------------------------------------------------------


class TriMulExecResult(BaseModel):
    """Raw outcome of running one candidate on one test case.

    ``score()`` returns ``Ok(TriMulExecResult)`` on successful execution of
    the scoring pipeline — even if the candidate itself was wrong. Only
    infrastructure failures (Modal crashes, etc.) produce ``Err``. Callers
    turn this into a reward / feedback by calling
    ``execution_feedback_from_exec_result``.
    """

    model_config = ConfigDict(frozen=True)

    correct: bool
    runtime_ns: float
    ref_runtime_ns: float
    error_message: str = ""
    runtime_error_name: str = ""
    runtime_error: str = ""
    traceback: str = ""
    compilation_error: str = ""
    failure_kind: Literal[
        "none", "compile_failed", "runtime_error", "incorrect"
    ] = "none"


def execution_feedback_from_exec_result(
    *, exec_result: TriMulExecResult, speedup: float
) -> TriMulKernelExecutionFeedback:
    """Adapt a raw ``TriMulExecResult`` to the feedback discriminated union."""
    match exec_result.failure_kind:
        case "compile_failed":
            return CompileFailedFeedback(
                compilation_error=exec_result.compilation_error,
            )
        case "runtime_error":
            return RuntimeErrorFeedback(
                runtime_error_name=exec_result.runtime_error_name,
                runtime_error=exec_result.runtime_error,
                traceback=exec_result.traceback,
            )
        case "incorrect":
            return IncorrectFeedback(error_message=exec_result.error_message)
        case "none":
            return SuccessFeedback(
                runtime_ns=exec_result.runtime_ns,
                ref_runtime_ns=exec_result.ref_runtime_ns,
                speedup=speedup,
            )
