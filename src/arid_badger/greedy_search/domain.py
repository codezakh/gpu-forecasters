from __future__ import annotations

from typing import Annotated, Dict, Iterable, Literal, Optional, Union

from kernelbench.eval import KernelExecResult
from pydantic import BaseModel, ConfigDict, Field
from ulid import ULID


class EvaluationMetrics(BaseModel):
    """Stable, JSON-friendly subset of KernelBench execution data."""

    model_config = ConfigDict(frozen=True)

    compiled: Optional[bool] = None
    correctness: Optional[bool] = None
    runtime: Optional[float] = None
    ref_runtime: Optional[float] = None


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


InvalidReason = Literal[
    "compile_failed",
    "incorrect",
    "nonpositive_runtime",
    "unknown",
]


class ValidEvaluation(BaseModel):
    """A valid evaluation; speedup exists and is meaningful."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["valid"] = "valid"
    speedup: float
    metrics: Optional[EvaluationMetrics] = None
    execution_feedback: KernelExecutionFeedback


class InvalidEvaluation(BaseModel):
    """An invalid evaluation; speedup is intentionally absent."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["invalid"] = "invalid"
    reason: InvalidReason
    metrics: Optional[EvaluationMetrics] = None
    execution_feedback: KernelExecutionFeedback


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


Evaluation = Annotated[
    Union[ValidEvaluation, InvalidEvaluation],
    Field(discriminator="kind"),
]


class KernelCandidate(BaseModel):
    """A candidate kernel considered by the search (entity; identity is ULID)."""

    model_config = ConfigDict(frozen=True)

    ulid: ULID = Field(default_factory=ULID)
    code: str = Field(min_length=1)
    parent_ulid: Optional[ULID] = None
    evaluation: Optional[Evaluation] = None


class CandidateGraph(BaseModel):
    """Graph of candidates keyed by identity (ULID).

    Note: Although JSON object keys must be strings, Pydantic can serialize ULID
    keys appropriately; we should keep ULIDs as ULIDs internally.
    """

    model_config = ConfigDict(frozen=True)

    candidates: Dict[ULID, KernelCandidate] = Field(default_factory=dict)

    def get(self, ulid: ULID) -> KernelCandidate:
        return self.candidates[ulid]

    def has(self, ulid: ULID) -> bool:
        return ulid in self.candidates

    def add(self, candidate: KernelCandidate) -> "CandidateGraph":
        updated = dict(self.candidates)
        updated[candidate.ulid] = candidate
        return self.model_copy(update={"candidates": updated})

    def with_evaluation(
        self, *, ulid: ULID, evaluation: Evaluation
    ) -> "CandidateGraph":
        existing = self.get(ulid)
        updated_candidate = existing.model_copy(update={"evaluation": evaluation})
        updated = dict(self.candidates)
        updated[ulid] = updated_candidate
        return self.model_copy(update={"candidates": updated})
