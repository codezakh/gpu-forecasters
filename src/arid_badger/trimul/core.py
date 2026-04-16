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


class CaseSpeedup(BaseModel):
    """Per-case speedup with the shape parameters that produced it.

    Gives mutation prompts actionable signal: "fast on small shapes,
    slow on large shapes" is only useful if the shape is attached.
    """

    model_config = ConfigDict(frozen=True)

    seqlen: int
    bs: int
    dim: int
    hiddendim: int
    nomask: bool
    distribution: str
    speedup: float
    runtime_ns: float
    ref_runtime_ns: float


class SuccessFeedback(BaseModel):
    """Candidate passed correctness on all cases; aggregated timings."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["success"] = "success"
    aggregated_speedup: float
    aggregation_method: str
    per_case_speedups: list[CaseSpeedup]


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
    ``failure_feedback_from_exec_result`` (for failures) or constructing
    ``SuccessFeedback`` directly (for the all-correct path).
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


TriMulFailureFeedback = Union[
    CompileFailedFeedback,
    RuntimeErrorFeedback,
    IncorrectFeedback,
]


def failure_feedback_from_exec_result(
    exec_result: TriMulExecResult,
) -> TriMulFailureFeedback:
    """Build a failure feedback from a failing ``TriMulExecResult``.

    Only valid when ``exec_result.failure_kind != "none"`` (i.e. the
    candidate did not pass).  The all-correct path builds
    ``SuccessFeedback`` directly in the provider layer, which has the
    full multi-case context needed for aggregation.
    """
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
            raise ValueError(
                "failure_feedback_from_exec_result called on a passing result "
                "(failure_kind='none'). Use SuccessFeedback directly for the "
                "all-correct path."
            )
