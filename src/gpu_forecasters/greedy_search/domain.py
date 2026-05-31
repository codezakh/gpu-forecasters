from __future__ import annotations

from typing import (
    Annotated,
    Dict,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Union,
)

from gpu_forecasters.typing_utils import Option

from pydantic import BaseModel, ConfigDict, Field
from ulid import ULID

from gpu_forecasters.kernelbench.core import KernelExecutionFeedback


class EvaluationMetrics(BaseModel):
    """Stable, JSON-friendly subset of KernelBench execution data."""

    model_config = ConfigDict(frozen=True)

    compiled: Optional[bool] = None
    correctness: Optional[bool] = None
    runtime: Optional[float] = None
    ref_runtime: Optional[float] = None


class MutationContext(BaseModel):
    """Context for mutating a kernel."""

    model_config = ConfigDict(frozen=True)

    # Reference architecture that KernelBench expects the generated code to preserve.
    reference_kernel_code: str = Field(min_length=1)
    previous_kernel_code: str = Field(min_length=1)
    previous_kernel_ulid: Optional[ULID] = None
    previous_evaluation: Optional["Evaluation"] = None
    num_mutations: int = 1
    backend: Literal["cuda", "triton"] = "cuda"
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"


class MutatedKernel(BaseModel):
    """Result of a kernel mutation."""

    model_config = ConfigDict(frozen=True)

    kernel_code: str = Field(min_length=1)
    ulid: ULID = Field(default_factory=ULID)
    ancestor_ulid: Optional[ULID] = None


class MutationFunction(Protocol):
    def __call__(self, context: MutationContext) -> MutatedKernel: ...



# Feedback types re-exported from gpu_forecasters.kernelbench.core:
# CompileFailedFeedback, RuntimeErrorFeedback, IncorrectFeedback,
# SuccessFeedback, KernelExecutionFeedback, execution_feedback_from_exec_result


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




Evaluation = Annotated[
    Union[ValidEvaluation, InvalidEvaluation],
    Field(discriminator="kind"),
]


# ---------------------------------------------------------------------------
# Attempt types (mutation / scoring outcomes)
# ---------------------------------------------------------------------------


class MutationError(BaseModel):
    model_config = ConfigDict(frozen=True)

    message: str
    exception_repr: str
    traceback: str


class ScoringError(BaseModel):
    model_config = ConfigDict(frozen=True)

    message: str
    exception_repr: str
    traceback: str


class MutationSuccess(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["success"] = "success"
    attempt_idx: int
    candidate_ulid: ULID


class MutationFailure(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["failure"] = "failure"
    attempt_idx: int
    error: MutationError


MutationAttempt = Annotated[
    Union[MutationSuccess, MutationFailure],
    Field(discriminator="kind"),
]


class ScoringSuccess(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["success"] = "success"
    candidate_ulid: ULID
    evaluation: Evaluation


class ScoringFailure(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["failure"] = "failure"
    candidate_ulid: ULID
    error: ScoringError


ScoringAttempt = Annotated[
    Union[ScoringSuccess, ScoringFailure],
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


# ---------------------------------------------------------------------------
# Provider protocols
# ---------------------------------------------------------------------------


class ScoringProvider(Protocol):
    def score_candidates(
        self,
        candidates: Sequence[KernelCandidate],
        reference_kernel_code: str,
    ) -> tuple[list[ScoringAttempt], list[tuple[KernelCandidate, Evaluation]]]: ...

    def score_reference(
        self, reference_kernel_code: str
    ) -> Option[ValidEvaluation, str]: ...


class MutationProvider(Protocol):
    def generate_mutations(
        self,
        context: MutationContext,
    ) -> tuple[list[MutationAttempt], list[KernelCandidate]]: ...
