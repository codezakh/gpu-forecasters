from __future__ import annotations

from typing import Annotated, Iterable, Literal, Union

from kernelbench.eval import KernelExecResult
from pydantic import BaseModel, ConfigDict, Field

from dataclasses import dataclass


@dataclass
class KernelScoringResult:
    """Result of scoring a kernel against a reference."""

    exec_result: KernelExecResult
    speedup: float
    is_valid: bool


# ---------------------------------------------------------------------------
# Execution feedback types
#
# A serialisation-safe, KernelBench-decoupled representation of what happened
# when a kernel was compiled and run.  These types extract just enough
# information from KernelExecResult to be useful for prompt construction
# without leaking third-party types into the rest of the codebase.
# ---------------------------------------------------------------------------


class CompileFailedFeedback(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["compile_failed"] = "compile_failed"
    compilation_error_name: str
    compilation_error: str


class RuntimeErrorFeedback(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["runtime_error"] = "runtime_error"
    runtime_error_name: str
    runtime_error: str
    runtime_error_traceback: str


class IncorrectFeedback(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["incorrect"] = "incorrect"
    correctness_issue: str
    max_difference: list[str]
    avg_difference: list[str]


class SuccessFeedback(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["success"] = "success"
    runtime_us: float
    ref_runtime_us: float
    speedup: float


KernelExecutionFeedback = Annotated[
    Union[
        CompileFailedFeedback,
        RuntimeErrorFeedback,
        IncorrectFeedback,
        SuccessFeedback,
    ],
    Field(discriminator="kind"),
]


class InfrastructureFailureFeedback(BaseModel):
    """Scoring subprocess crashed or timed out — no KernelExecResult available."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["infrastructure_failure"] = "infrastructure_failure"
    reason: str


def _stringify_metadata_value(value: object) -> str:
    if value is None:
        return ""
    return str(value)


def _stringify_metadata_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
        return [str(item) for item in value]
    return [str(value)]


def execution_feedback_from_exec_result(
    *, exec_result: KernelExecResult, speedup: float, is_valid: bool
) -> KernelExecutionFeedback:
    metadata = exec_result.metadata or {}
    if exec_result.compiled is False:
        return CompileFailedFeedback(
            compilation_error_name=_stringify_metadata_value(
                metadata.get("compilation_error_name")
            ),
            compilation_error=_stringify_metadata_value(
                metadata.get("compilation_error")
            ),
        )

    if exec_result.correctness is False:
        if (
            "runtime_error" in metadata
            or "runtime_error_name" in metadata
            or "runtime_error_traceback" in metadata
        ):
            return RuntimeErrorFeedback(
                runtime_error_name=_stringify_metadata_value(
                    metadata.get("runtime_error_name")
                ),
                runtime_error=_stringify_metadata_value(metadata.get("runtime_error")),
                runtime_error_traceback=_stringify_metadata_value(
                    metadata.get("runtime_error_traceback")
                ),
            )

        return IncorrectFeedback(
            correctness_issue=_stringify_metadata_value(
                metadata.get("correctness_issue")
            ),
            max_difference=_stringify_metadata_list(metadata.get("max_difference")),
            avg_difference=_stringify_metadata_list(metadata.get("avg_difference")),
        )

    if is_valid:
        return SuccessFeedback(
            runtime_us=exec_result.runtime,
            ref_runtime_us=exec_result.ref_runtime,
            speedup=speedup,
        )

    return IncorrectFeedback(
        correctness_issue="Evaluation marked invalid despite correctness=True.",
        max_difference=[],
        avg_difference=[],
    )
