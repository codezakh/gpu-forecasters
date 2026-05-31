from __future__ import annotations

from typing import List, Sequence, Tuple

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, computed_field
from ulid import ULID

from .domain import (
    CandidateGraph,
    Evaluation,
    KernelCandidate,
)
from .trace import RoundTrace, SearchTrace


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

    def validate_invariants(self) -> None:
        """Validate internal consistency of the checkpoint.

        This is not a guarantee the checkpoint is uncorrupted, but it catches
        common inconsistencies early and produces a clearer error than a later
        KeyError inside the search loop.
        """

        if self.cursor.next_depth < 0:
            raise ValueError(
                f"Invalid checkpoint: cursor.next_depth < 0 ({self.cursor.next_depth})"
            )

        if self.cursor.next_depth != len(self.trace.rounds):
            raise ValueError(
                "Invalid checkpoint: cursor.next_depth does not match trace length "
                f"(next_depth={self.cursor.next_depth}, rounds={len(self.trace.rounds)})"
            )

        if not self.candidates.has(self.cursor.parent_ulid):
            raise ValueError(
                "Invalid checkpoint: cursor.parent_ulid not found in candidates "
                f"({self.cursor.parent_ulid})"
            )

        if not self.candidates.has(self.best_ulid):
            raise ValueError(
                "Invalid checkpoint: best_ulid not found in candidates "
                f"({self.best_ulid})"
            )

        best_candidate = self.candidates.get(self.best_ulid)
        if best_candidate.evaluation is None:
            raise ValueError(
                "Invalid checkpoint: best candidate has no evaluation "
                f"({self.best_ulid})"
            )

        if best_candidate.evaluation != self.best_evaluation:
            raise ValueError(
                "Invalid checkpoint: best_evaluation does not match best candidate evaluation "
                f"({self.best_ulid})"
            )

    @computed_field
    @property
    def rounds(self) -> List[RoundTrace]:
        return self.trace.rounds

    def best_candidate(self) -> KernelCandidate:
        return self.candidates.get(self.best_ulid)

    def current_parent(self) -> KernelCandidate:
        """Return the current parent candidate for the next mutation round."""
        return self.candidates.get(self.cursor.parent_ulid)

    def evaluated_candidates(self) -> List[KernelCandidate]:
        return [
            c for c in self.candidates.candidates.values() if c.evaluation is not None
        ]

    def register_generated_candidates(
        self, generated: Sequence[KernelCandidate]
    ) -> None:
        """Add newly generated candidates to the graph (unscored)."""
        logger.debug(
            "Checkpoint update: registering {count} generated candidate(s).",
            count=len(generated),
        )
        graph = self.candidates
        for candidate in generated:
            graph = graph.add(candidate)
        self.candidates = graph

    def register_scored_candidates(
        self, scored: Sequence[Tuple[KernelCandidate, Evaluation]]
    ) -> None:
        """Attach evaluations for candidates that were successfully scored."""
        logger.debug(
            "Checkpoint update: registering {count} scored candidate(s).",
            count=len(scored),
        )
        graph = self.candidates
        for candidate, evaluation in scored:
            graph = graph.with_evaluation(ulid=candidate.ulid, evaluation=evaluation)
        self.candidates = graph

    def set_best(self, *, ulid: ULID) -> None:
        """Update best candidate pointer and cached evaluation (state update only)."""
        if not self.candidates.has(ulid):
            raise ValueError(
                "Cannot set best candidate: ULID not found in candidates " f"({ulid})"
            )
        candidate = self.candidates.get(ulid)
        if candidate.evaluation is None:
            raise ValueError(
                "Cannot set best candidate: candidate has no evaluation " f"({ulid})"
            )
        self.best_ulid = ulid
        self.best_evaluation = candidate.evaluation

    def set_best_candidate(self, *, candidate: KernelCandidate) -> None:
        """Update best candidate pointer and cached evaluation from an entity."""
        if not self.candidates.has(candidate.ulid):
            raise ValueError(
                "Cannot set best candidate: candidate not found in candidates "
                f"({candidate.ulid})"
            )
        if candidate.evaluation is None:
            raise ValueError(
                "Cannot set best candidate: candidate has no evaluation "
                f"({candidate.ulid})"
            )
        self.best_ulid = candidate.ulid
        self.best_evaluation = candidate.evaluation

    def append_round_trace(self, round_trace: RoundTrace) -> None:
        """Append a completed round to the search trace."""
        self.trace = self.trace.model_copy(
            update={"rounds": [*self.trace.rounds, round_trace]}
        )
        logger.debug(
            "Checkpoint update: appended round trace (rounds={rounds}).",
            rounds=len(self.trace.rounds),
        )

    def advance_cursor(self, *, next_depth: int, parent_ulid: ULID) -> None:
        """Advance cursor to the next round."""
        self.cursor = SearchCursor(next_depth=next_depth, parent_ulid=parent_ulid)
        logger.debug(
            "Checkpoint update: advanced cursor to depth {next_depth} (parent_ulid={parent_ulid}).",
            next_depth=next_depth,
            parent_ulid=parent_ulid,
        )

    def advance_parent(self, *, next_depth: int, parent: KernelCandidate) -> None:
        """Advance cursor using a parent candidate entity (hides ULID plumbing)."""
        self.advance_cursor(next_depth=next_depth, parent_ulid=parent.ulid)
