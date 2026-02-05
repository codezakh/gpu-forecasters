from __future__ import annotations

from typing import Annotated, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, computed_field
from ulid import ULID


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


class FoundRoundBest(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["found"] = "found"
    best_candidate_ulid: ULID
    best_evaluation: ValidEvaluation
    num_scored: int
    num_valid: int

    @computed_field
    @property
    def best_speedup(self) -> float:
        return self.best_evaluation.speedup


class NoScoredRoundBest(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["no_scored"] = "no_scored"
    num_scored: int = 0
    num_valid: int = 0


class AllInvalidRoundBest(BaseModel):
    model_config = ConfigDict(frozen=True)

    kind: Literal["all_invalid"] = "all_invalid"
    num_scored: int
    num_valid: int = 0


RoundBest = Annotated[
    Union[FoundRoundBest, NoScoredRoundBest, AllInvalidRoundBest],
    Field(discriminator="kind"),
]


class RoundTrace(BaseModel):
    model_config = ConfigDict(frozen=True)

    depth: int
    parent_ulid: ULID

    mutation_attempts: List[MutationAttempt]
    scoring_attempts: List[ScoringAttempt]

    round_best: RoundBest
    selected_parent_ulid: ULID


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


class SearchCursor(BaseModel):
    model_config = ConfigDict(frozen=True)

    next_depth: int
    parent_ulid: ULID


class SearchTrace(BaseModel):
    model_config = ConfigDict(frozen=True)

    rounds: List[RoundTrace] = Field(default_factory=list)


class GreedySearchCheckpoint(BaseModel):
    """Checkpointable state of a greedy search run (with full trace)."""

    model_config = ConfigDict(frozen=True)

    cursor: SearchCursor
    candidates: CandidateGraph

    best_ulid: ULID
    best_evaluation: Evaluation

    trace: SearchTrace = Field(default_factory=SearchTrace)

    @computed_field
    @property
    def rounds(self) -> List[RoundTrace]:
        return self.trace.rounds

    def best_candidate(self) -> KernelCandidate:
        return self.candidates.get(self.best_ulid)

    def evaluated_candidates(self) -> List[KernelCandidate]:
        return [
            c for c in self.candidates.candidates.values() if c.evaluation is not None
        ]


# Backwards-compatible naming (temporary; callers should migrate).
GreedySearchResult = GreedySearchCheckpoint
