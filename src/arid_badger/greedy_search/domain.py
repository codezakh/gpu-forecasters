from __future__ import annotations

from typing import Annotated, Dict, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field
from ulid import ULID


class EvaluationMetrics(BaseModel):
    """Stable, JSON-friendly subset of KernelBench execution data."""

    model_config = ConfigDict(frozen=True)

    compiled: Optional[bool] = None
    correctness: Optional[bool] = None
    runtime: Optional[float] = None
    ref_runtime: Optional[float] = None


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


class InvalidEvaluation(BaseModel):
    """An invalid evaluation; speedup is intentionally absent."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["invalid"] = "invalid"
    reason: InvalidReason
    metrics: Optional[EvaluationMetrics] = None


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
    """Graph of candidates keyed by identity (stored as string keys for JSON)."""

    model_config = ConfigDict(frozen=True)

    candidates: Dict[str, KernelCandidate] = Field(default_factory=dict)

    def get(self, ulid: ULID) -> KernelCandidate:
        return self.candidates[str(ulid)]

    def has(self, ulid: ULID) -> bool:
        return str(ulid) in self.candidates

    def add(self, candidate: KernelCandidate) -> "CandidateGraph":
        updated = dict(self.candidates)
        updated[str(candidate.ulid)] = candidate
        return self.model_copy(update={"candidates": updated})

    def with_evaluation(
        self, *, ulid: ULID, evaluation: Evaluation
    ) -> "CandidateGraph":
        existing = self.get(ulid)
        updated_candidate = existing.model_copy(update={"evaluation": evaluation})
        updated = dict(self.candidates)
        updated[str(ulid)] = updated_candidate
        return self.model_copy(update={"candidates": updated})
