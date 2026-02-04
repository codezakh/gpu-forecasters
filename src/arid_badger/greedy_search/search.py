from __future__ import annotations

import traceback
from typing import Callable, List, Literal, Optional, Tuple

from ulid import ULID

from .components import MutationContext, MutationFunction, MutatedKernel
from arid_badger.kernelbench.core import KernelScoringResult
from arid_badger.typing_utils import Option
import attrs
from loguru import logger

from .outcomes import (
    AllInvalidRoundBest,
    GreedySearchResult,
    FoundRoundBest,
    MutationAttemptTrace,
    MutationError,
    NoScoredRoundBest,
    RoundBest,
    RoundReport,
    ScoringAttemptTrace,
    ScoringError,
)


@attrs.define
class GreedySearchConfig:
    """Configuration for greedy search."""

    max_depth: int
    num_mutations: int
    starter_kernel_code: str
    reference_kernel_code: str
    mutation_function: MutationFunction
    scoring_function: Callable[[str, str], KernelScoringResult]
    backend: Literal["cuda", "triton"] = "cuda"
    precision: Literal["fp32", "fp16", "bf16"] = "fp32"
    prompt_option: Literal["zero_shot", "one_shot", "few_shot"] = "one_shot"
    model_slug: str = "gemini/gemini-3-flash-preview"


@attrs.define
class GreedySearchState:
    """State of the greedy search during execution."""

    current_best_kernel_code: str
    current_best_kernel_ulid: Optional[ULID]
    best_kernel: MutatedKernel
    best_score: KernelScoringResult
    search_history: List[Tuple[MutatedKernel, KernelScoringResult]]
    rounds: List[RoundReport]


class GreedySearch:
    def __init__(self, config: GreedySearchConfig):
        """
        Initialize a GreedySearch instance.

        Args:
            config: Search configuration containing all search parameters and dependencies
        """
        self.config = config

    def _attempt_generate_mutations(
        self, *, parent_kernel_code: str, parent_kernel_ulid: Optional[ULID]
    ) -> Tuple[List[MutationAttemptTrace], List[MutatedKernel]]:
        mutation_context = MutationContext(
            reference_kernel_code=self.config.reference_kernel_code,
            previous_kernel_code=parent_kernel_code,
            previous_kernel_ulid=parent_kernel_ulid,
            backend=self.config.backend,
            precision=self.config.precision,
            prompt_option=self.config.prompt_option,
            model_slug=self.config.model_slug,
        )

        attempts: List[MutationAttemptTrace] = []
        generated: List[MutatedKernel] = []
        for attempt_idx in range(self.config.num_mutations):
            try:
                mutated = self.config.mutation_function(mutation_context)
                attempts.append(
                    MutationAttemptTrace(
                        attempt_idx=attempt_idx, result=Option.ok(mutated)
                    )
                )
                generated.append(mutated)
            except Exception as e:
                attempts.append(
                    MutationAttemptTrace(
                        attempt_idx=attempt_idx,
                        result=Option.err(
                            MutationError(
                                message="Mutation function raised an exception",
                                exception_repr=repr(e),
                                traceback=traceback.format_exc(),
                            )
                        ),
                    )
                )

        return attempts, generated

    def _attempt_score_mutations(
        self, mutations: List[MutatedKernel]
    ) -> Tuple[
        List[ScoringAttemptTrace], List[Tuple[MutatedKernel, KernelScoringResult]]
    ]:
        attempts: List[ScoringAttemptTrace] = []
        scored: List[Tuple[MutatedKernel, KernelScoringResult]] = []
        for mutation in mutations:
            try:
                score = self.config.scoring_function(
                    mutation.kernel_code, self.config.reference_kernel_code
                )
                pair = (mutation, score)
                attempts.append(
                    ScoringAttemptTrace(
                        mutation_ulid=mutation.ulid, result=Option.ok(pair)
                    )
                )
                scored.append(pair)
            except Exception as e:
                attempts.append(
                    ScoringAttemptTrace(
                        mutation_ulid=mutation.ulid,
                        result=Option.err(
                            ScoringError(
                                message="Scoring function raised an exception",
                                exception_repr=repr(e),
                                traceback=traceback.format_exc(),
                            )
                        ),
                    )
                )

        return attempts, scored

    def _select_round_best(
        self, scored_mutations: List[Tuple[MutatedKernel, KernelScoringResult]]
    ) -> RoundBest:
        if not scored_mutations:
            return NoScoredRoundBest()

        valid = [(m, s) for (m, s) in scored_mutations if s.is_valid]
        if not valid:
            return AllInvalidRoundBest(num_scored=len(scored_mutations))

        best_mutation, best_score = max(valid, key=lambda ms: ms[1].speedup)
        return FoundRoundBest(
            best_mutation=best_mutation,
            best_score=best_score,
            num_scored=len(scored_mutations),
            num_valid=len(valid),
        )

    def search(self) -> GreedySearchResult:
        """
        Perform greedy search for optimized kernels.

        Returns:
            GreedySearchResult with best kernel, best score, and full search history
        """
        # Score the starter kernel to establish baseline
        starter_score = self.config.scoring_function(
            self.config.starter_kernel_code, self.config.reference_kernel_code
        )
        # Create a MutatedKernel for the starter (no ancestor)
        starter_mutated = MutatedKernel(
            kernel_code=self.config.starter_kernel_code,
            ancestor_ulid=None,
        )

        # Initialize search state
        state = GreedySearchState(
            current_best_kernel_code=self.config.starter_kernel_code,
            current_best_kernel_ulid=None,
            best_kernel=starter_mutated,
            best_score=starter_score,
            search_history=[(starter_mutated, starter_score)],
            rounds=[],
        )

        # Iterate through depth levels (always bounded by max_depth)
        for depth in range(self.config.max_depth):
            mutation_attempts, mutations = self._attempt_generate_mutations(
                parent_kernel_code=state.current_best_kernel_code,
                parent_kernel_ulid=state.current_best_kernel_ulid,
            )
            scoring_attempts, scored_mutations = self._attempt_score_mutations(
                mutations
            )

            round_best = self._select_round_best(scored_mutations)

            global_best_updated = False
            # Parent update policy:
            # - If a valid best-of-round exists: mutate it next round (KernelBench spec)
            # - Otherwise: keep the same parent and continue (explicit policy)
            match round_best:
                case FoundRoundBest() as found:
                    if found.best_speedup > state.best_score.speedup:
                        state.best_kernel = found.best_mutation
                        state.best_score = found.best_score
                        global_best_updated = True

                    next_parent_code = found.best_mutation.kernel_code
                    next_parent_ulid = found.best_mutation.ulid
                case NoScoredRoundBest() | AllInvalidRoundBest():
                    next_parent_code = state.current_best_kernel_code
                    next_parent_ulid = state.current_best_kernel_ulid

            # Always record round report + history (full trace)
            state.rounds.append(
                RoundReport(
                    depth=depth,
                    parent_kernel_ulid=state.current_best_kernel_ulid,
                    parent_kernel_code=state.current_best_kernel_code,
                    mutation_attempts=mutation_attempts,
                    scoring_attempts=scoring_attempts,
                    scored_mutations=scored_mutations,
                    round_best=round_best,
                    global_best_updated=global_best_updated,
                    next_parent_ulid=next_parent_ulid,
                    next_parent_code=next_parent_code,
                )
            )
            state.search_history.extend(scored_mutations)

            logger.info(
                "GreedySearch depth={depth} parent_ulid={parent_ulid} "
                "mut_attempts={mut_attempts} mut_ok={mut_ok} "
                "score_attempts={score_attempts} scored={scored} valid={valid} "
                "round_best={round_best} global_best_updated={global_best_updated}",
                depth=depth,
                parent_ulid=(
                    str(state.current_best_kernel_ulid)
                    if state.current_best_kernel_ulid is not None
                    else None
                ),
                mut_attempts=len(mutation_attempts),
                mut_ok=len(mutations),
                score_attempts=len(scoring_attempts),
                scored=len(scored_mutations),
                valid=round_best.num_valid,
                round_best=round_best.kind,
                global_best_updated=global_best_updated,
            )

            state.current_best_kernel_code = next_parent_code
            state.current_best_kernel_ulid = next_parent_ulid

        return GreedySearchResult(
            best_kernel=state.best_kernel,
            best_score=state.best_score,
            rounds=state.rounds,
            search_history=state.search_history,
        )
