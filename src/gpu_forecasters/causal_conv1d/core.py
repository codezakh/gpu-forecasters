"""Causal conv1d execution feedback types and exec result.

Near-duplicate of ``arid_badger.trimul.core``. The feedback union, exec
result envelope, infrastructure-failure variant, and
``failure_feedback_from_exec_result`` are entirely kernel-agnostic and
will be lifted into ``arid_badger.gpu_mode_kernel.feedback`` in the
gh070-A task #3 extraction.

The only thing that differs from TriMul is ``CaseSpeedup``'s shape
fields: ``B, D, S, W`` here vs ``seqlen, bs, dim, hiddendim, nomask,
distribution`` there. That difference is the kernel-specific seam.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Adaptive-loop stats (identical to TriMul; will be hoisted in #3).
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
    model_config = ConfigDict(frozen=True)

    kind: Literal["compile_failed"] = "compile_failed"
    compilation_error: str


class RuntimeErrorFeedback(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["runtime_error"] = "runtime_error"
    runtime_error_name: str
    runtime_error: str
    traceback: str


class IncorrectFeedback(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["incorrect"] = "incorrect"
    error_message: str


class CaseSpeedup(BaseModel):
    """Per-case speedup with the shape parameters that produced it.

    Shape fields (B, D, S, W) are the kernel-specific seam — TriMul's
    sibling carries (seqlen, bs, dim, hiddendim, nomask, distribution)
    instead. Everything else (runtime, ref runtime, speedup) is generic.
    """

    model_config = ConfigDict(frozen=True)

    B: int
    D: int
    S: int
    W: int
    speedup: float
    runtime_ns: float
    ref_runtime_ns: float


class SuccessFeedback(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["success"] = "success"
    aggregated_speedup: float
    aggregation_method: str
    per_case_speedups: list[CaseSpeedup]


class InfrastructureFailureFeedback(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["infrastructure_failure"] = "infrastructure_failure"
    reason: str


CausalConv1dKernelExecutionFeedback = Annotated[
    Union[
        CompileFailedFeedback,
        RuntimeErrorFeedback,
        IncorrectFeedback,
        SuccessFeedback,
    ],
    Field(discriminator="kind"),
]


# ---------------------------------------------------------------------------
# Exec result — wire-format return type of ``score()``
# ---------------------------------------------------------------------------


class CausalConv1dExecResult(BaseModel):
    """Raw outcome of running one candidate on one test case."""

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


CausalConv1dFailureFeedback = Union[
    CompileFailedFeedback,
    RuntimeErrorFeedback,
    IncorrectFeedback,
]


def failure_feedback_from_exec_result(
    exec_result: CausalConv1dExecResult,
) -> CausalConv1dFailureFeedback:
    """Build a failure feedback from a failing exec result.

    Only valid when ``exec_result.failure_kind != "none"``. The
    all-correct path builds ``SuccessFeedback`` directly in the provider
    layer, which has the per-case context needed for aggregation.
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
                "failure_feedback_from_exec_result called on a passing "
                "result (failure_kind='none'). Use SuccessFeedback "
                "directly for the all-correct path."
            )
