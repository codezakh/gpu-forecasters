from __future__ import annotations

from typing import List, Literal, Optional, Tuple, Union

import attrs
from ulid import ULID

from arid_badger.kernelbench.core import KernelScoringResult
from arid_badger.typing_utils import Option

from .components import MutatedKernel


@attrs.define(frozen=True, slots=True)
class MutationError:
    message: str
    exception_repr: str
    traceback: str


@attrs.define(frozen=True, slots=True)
class ScoringError:
    message: str
    exception_repr: str
    traceback: str


@attrs.define(frozen=True, slots=True)
class MutationAttemptTrace:
    attempt_idx: int
    result: Option[MutatedKernel, MutationError]


@attrs.define(frozen=True, slots=True)
class ScoringAttemptTrace:
    mutation_ulid: ULID
    result: Option[Tuple[MutatedKernel, KernelScoringResult], ScoringError]


@attrs.define(frozen=True, slots=True)
class FoundRoundBest:
    best_mutation: MutatedKernel
    best_score: KernelScoringResult
    num_scored: int
    num_valid: int
    kind: Literal["found"] = "found"

    @property
    def best_speedup(self) -> float:
        return self.best_score.speedup


@attrs.define(frozen=True, slots=True)
class NoScoredRoundBest:
    num_scored: int = 0
    num_valid: int = 0
    kind: Literal["no_scored"] = "no_scored"


@attrs.define(frozen=True, slots=True)
class AllInvalidRoundBest:
    num_scored: int
    num_valid: int = 0
    kind: Literal["all_invalid"] = "all_invalid"


RoundBest = Union[FoundRoundBest, NoScoredRoundBest, AllInvalidRoundBest]


@attrs.define(frozen=True, slots=True)
class RoundReport:
    depth: int

    parent_kernel_ulid: Optional[ULID]
    parent_kernel_code: str

    mutation_attempts: List[MutationAttemptTrace]
    scoring_attempts: List[ScoringAttemptTrace]
    scored_mutations: List[Tuple[MutatedKernel, KernelScoringResult]]

    round_best: RoundBest

    global_best_updated: bool
    next_parent_ulid: Optional[ULID]
    next_parent_code: str


@attrs.define(frozen=True, slots=True)
class GreedySearchResult:
    """Result of a greedy search run (including full per-round traces)."""

    best_kernel: MutatedKernel
    best_score: KernelScoringResult
    rounds: List[RoundReport]

    # Convenience: flat list of successfully-scored (kernel, score) pairs, including
    # the starter kernel. Source-of-truth is `rounds`.
    search_history: List[Tuple[MutatedKernel, KernelScoringResult]]
