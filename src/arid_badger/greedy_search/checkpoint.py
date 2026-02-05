from __future__ import annotations

from typing import List

from pydantic import BaseModel, ConfigDict, Field, computed_field
from ulid import ULID

from .domain import CandidateGraph, Evaluation, KernelCandidate
from .trace import RoundTrace, SearchTrace


class SearchCursor(BaseModel):
    model_config = ConfigDict(frozen=True)

    next_depth: int
    parent_ulid: ULID


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
