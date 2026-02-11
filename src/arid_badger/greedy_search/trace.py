from __future__ import annotations

from typing import Annotated, List, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, computed_field
from ulid import ULID

from .domain import (
    Evaluation,
    MutationAttempt,
    MutationError,
    MutationFailure,
    MutationSuccess,
    ScoringAttempt,
    ScoringError,
    ScoringFailure,
    ScoringSuccess,
    ValidEvaluation,
)

# Re-export attempt types so existing consumers continue to work.
__all__ = [
    "MutationError",
    "ScoringError",
    "MutationSuccess",
    "MutationFailure",
    "MutationAttempt",
    "ScoringSuccess",
    "ScoringFailure",
    "ScoringAttempt",
    "RoundWinnerSelected",
    "RoundNoEvaluations",
    "RoundAllEvaluationsInvalid",
    "RoundOutcome",
    "RoundTrace",
    "SearchTrace",
]


class RoundWinnerSelected(BaseModel):
    """This round produced at least one valid evaluation and selected a winner."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["winner_selected"] = "winner_selected"
    winner_ulid: ULID
    winner_evaluation: ValidEvaluation
    num_scored: int
    num_valid: int

    @computed_field
    @property
    def winner_speedup(self) -> float:
        return self.winner_evaluation.speedup


class RoundNoEvaluations(BaseModel):
    """No evaluations were produced (e.g., no candidates generated or scoring all raised)."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["no_evaluations"] = "no_evaluations"

    @computed_field
    @property
    def num_scored(self) -> int:
        return 0

    @computed_field
    @property
    def num_valid(self) -> int:
        return 0


class RoundAllEvaluationsInvalid(BaseModel):
    """Evaluations were produced, but none were valid."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["all_evaluations_invalid"] = "all_evaluations_invalid"
    num_scored: int

    @computed_field
    @property
    def num_valid(self) -> int:
        return 0


RoundOutcome = Annotated[
    Union[RoundWinnerSelected, RoundNoEvaluations, RoundAllEvaluationsInvalid],
    Field(discriminator="kind"),
]


class RoundTrace(BaseModel):
    model_config = ConfigDict(frozen=True)

    depth: int
    parent_ulid: ULID

    mutation_attempts: List[MutationAttempt]
    scoring_attempts: List[ScoringAttempt]

    outcome: RoundOutcome
    selected_parent_ulid: ULID


class SearchTrace(BaseModel):
    model_config = ConfigDict(frozen=True)

    rounds: List[RoundTrace] = Field(default_factory=list)
