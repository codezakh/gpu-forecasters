from __future__ import annotations

from typing import List, Sequence, Tuple, cast

from pydantic import BaseModel, ConfigDict, Field, computed_field
from ulid import ULID

from .domain import (
    CandidateGraph,
    Evaluation,
    InvalidEvaluation,
    KernelCandidate,
    ValidEvaluation,
)
from .trace import RoundOutcome, RoundTrace, RoundWinnerSelected, SearchTrace


class SearchCursor(BaseModel):
    model_config = ConfigDict(frozen=True)

    next_depth: int
    parent_ulid: ULID


class GreedySearchCheckpoint(BaseModel):
    """Checkpointable state of a greedy search run (with full trace)."""

    model_config = ConfigDict(validate_assignment=True)

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

    def register_generated_candidates(self, generated: Sequence[KernelCandidate]) -> None:
        """Add newly generated candidates to the graph (unscored)."""
        graph = self.candidates
        for candidate in generated:
            graph = graph.add(candidate)
        self.candidates = graph

    def register_scored_candidates(
        self, scored: Sequence[Tuple[KernelCandidate, Evaluation]]
    ) -> None:
        """Attach evaluations for candidates that were successfully scored."""
        graph = self.candidates
        for candidate, evaluation in scored:
            graph = graph.with_evaluation(ulid=candidate.ulid, evaluation=evaluation)
        self.candidates = graph

    def select_parent_ulid(self, outcome: RoundOutcome) -> ULID:
        """Choose the next parent candidate ULID given the round outcome."""
        if outcome.kind == "winner_selected":
            winner = cast(RoundWinnerSelected, outcome)
            return winner.winner_ulid
        return self.cursor.parent_ulid

    def update_best_from_outcome(self, outcome: RoundOutcome) -> None:
        """Update global best candidate/evaluation using the round outcome."""
        if outcome.kind != "winner_selected":
            return

        winner = cast(RoundWinnerSelected, outcome)

        if isinstance(self.best_evaluation, InvalidEvaluation):
            self.best_ulid = winner.winner_ulid
            self.best_evaluation = winner.winner_evaluation
            return

        # Only update if we have a meaningful comparison.
        if isinstance(self.best_evaluation, ValidEvaluation):
            if winner.winner_speedup > self.best_evaluation.speedup:
                self.best_ulid = winner.winner_ulid
                self.best_evaluation = winner.winner_evaluation

    def append_round_trace(self, round_trace: RoundTrace) -> None:
        """Append a completed round to the search trace."""
        self.trace = self.trace.model_copy(
            update={"rounds": [*self.trace.rounds, round_trace]}
        )

    def advance_cursor(self, *, next_depth: int, parent_ulid: ULID) -> None:
        """Advance cursor to the next round."""
        self.cursor = SearchCursor(next_depth=next_depth, parent_ulid=parent_ulid)
