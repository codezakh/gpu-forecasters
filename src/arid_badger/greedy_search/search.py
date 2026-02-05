from __future__ import annotations

import traceback
from typing import Callable, List, Literal, Tuple

from .components import MutationContext, MutationFunction
from arid_badger.kernelbench.core import KernelScoringResult
import attrs
from loguru import logger

from .checkpoint import GreedySearchCheckpoint, SearchCursor
from .domain import CandidateGraph, Evaluation, EvaluationMetrics, KernelCandidate
from .trace import (
    MutationAttempt,
    MutationFailure,
    MutationError,
    MutationSuccess,
    RoundAllEvaluationsInvalid,
    RoundNoEvaluations,
    RoundOutcome,
    RoundTrace,
    RoundWinnerSelected,
    ScoringAttempt,
    ScoringError,
    ScoringFailure,
    ScoringSuccess,
    SearchTrace,
)
from .domain import InvalidEvaluation, ValidEvaluation


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


class GreedySearch:
    def __init__(self, config: GreedySearchConfig):
        """
        Initialize a GreedySearch instance.

        Args:
            config: Search configuration containing all search parameters and dependencies
        """
        self.config = config

    def _score_to_evaluation(self, score: KernelScoringResult) -> Evaluation:
        exec_result = score.exec_result
        metrics = EvaluationMetrics(
            compiled=getattr(exec_result, "compiled", None),
            correctness=getattr(exec_result, "correctness", None),
            runtime=getattr(exec_result, "runtime", None),
            ref_runtime=getattr(exec_result, "ref_runtime", None),
        )

        if score.is_valid:
            return ValidEvaluation(speedup=score.speedup, metrics=metrics)

        reason: str = "unknown"
        if metrics.compiled is False:
            reason = "compile_failed"
        elif metrics.correctness is False:
            reason = "incorrect"
        elif (
            metrics.runtime is not None
            and metrics.ref_runtime is not None
            and (metrics.runtime <= 0 or metrics.ref_runtime <= 0)
        ):
            reason = "nonpositive_runtime"
        return InvalidEvaluation(reason=reason, metrics=metrics)

    def _attempt_generate_mutations(
        self, *, parent: KernelCandidate
    ) -> Tuple[List[MutationAttempt], List[KernelCandidate]]:
        mutation_context = MutationContext(
            reference_kernel_code=self.config.reference_kernel_code,
            previous_kernel_code=parent.code,
            previous_kernel_ulid=parent.ulid,
            backend=self.config.backend,
            precision=self.config.precision,
            prompt_option=self.config.prompt_option,
            model_slug=self.config.model_slug,
        )

        attempts: List[MutationAttempt] = []
        generated: List[KernelCandidate] = []
        for attempt_idx in range(self.config.num_mutations):
            try:
                mutated = self.config.mutation_function(mutation_context)
                candidate = KernelCandidate(
                    ulid=mutated.ulid,
                    code=mutated.kernel_code,
                    parent_ulid=mutated.ancestor_ulid,
                    evaluation=None,
                )
                attempts.append(
                    MutationSuccess(
                        attempt_idx=attempt_idx, candidate_ulid=candidate.ulid
                    )
                )
                generated.append(candidate)
            except Exception as e:
                attempts.append(
                    MutationFailure(
                        attempt_idx=attempt_idx,
                        error=MutationError(
                            message="Mutation function raised an exception",
                            exception_repr=repr(e),
                            traceback=traceback.format_exc(),
                        ),
                    )
                )

        return attempts, generated

    def _attempt_score_mutations(
        self, candidates: List[KernelCandidate]
    ) -> Tuple[List[ScoringAttempt], List[Tuple[KernelCandidate, Evaluation]]]:
        attempts: List[ScoringAttempt] = []
        scored: List[Tuple[KernelCandidate, Evaluation]] = []
        for candidate in candidates:
            try:
                raw_score = self.config.scoring_function(
                    candidate.code, self.config.reference_kernel_code
                )
                evaluation = self._score_to_evaluation(raw_score)
                pair = (candidate, evaluation)
                attempts.append(
                    ScoringSuccess(candidate_ulid=candidate.ulid, evaluation=evaluation)
                )
                scored.append(pair)
            except Exception as e:
                attempts.append(
                    ScoringFailure(
                        candidate_ulid=candidate.ulid,
                        error=ScoringError(
                            message="Scoring function raised an exception",
                            exception_repr=repr(e),
                            traceback=traceback.format_exc(),
                        ),
                    )
                )

        return attempts, scored

    def _select_round_outcome(
        self, scored_candidates: List[Tuple[KernelCandidate, Evaluation]]
    ) -> RoundOutcome:
        if not scored_candidates:
            return RoundNoEvaluations()

        valid: List[Tuple[KernelCandidate, ValidEvaluation]] = []
        for candidate, evaluation in scored_candidates:
            if isinstance(evaluation, ValidEvaluation):
                valid.append((candidate, evaluation))

        if not valid:
            return RoundAllEvaluationsInvalid(num_scored=len(scored_candidates))

        best_candidate, best_evaluation = max(valid, key=lambda ce: ce[1].speedup)
        return RoundWinnerSelected(
            winner_ulid=best_candidate.ulid,
            winner_evaluation=best_evaluation,
            num_scored=len(scored_candidates),
            num_valid=len(valid),
        )

    def search(self) -> GreedySearchCheckpoint:
        """
        Perform greedy search for optimized kernels.

        Returns:
            GreedySearchCheckpoint sufficient to resume search.
        """
        # Score the starter kernel to establish baseline
        starter_raw_score = self.config.scoring_function(
            self.config.starter_kernel_code, self.config.reference_kernel_code
        )
        starter_evaluation = self._score_to_evaluation(starter_raw_score)

        starter_candidate = KernelCandidate(
            code=self.config.starter_kernel_code,
            parent_ulid=None,
            evaluation=starter_evaluation,
        )
        candidates = CandidateGraph().add(starter_candidate)

        checkpoint = GreedySearchCheckpoint(
            cursor=SearchCursor(next_depth=0, parent_ulid=starter_candidate.ulid),
            candidates=candidates,
            best_ulid=starter_candidate.ulid,
            best_evaluation=starter_evaluation,
            trace=SearchTrace(rounds=[]),
        )

        # Iterate through depth levels (always bounded by max_depth)
        for depth in range(checkpoint.cursor.next_depth, self.config.max_depth):
            current_parent_ulid = checkpoint.cursor.parent_ulid
            parent = checkpoint.candidates.get(current_parent_ulid)

            mutation_attempts, generated_candidates = self._attempt_generate_mutations(
                parent=parent
            )

            # Add generated candidates to the graph immediately so we can reference them by ULID.
            checkpoint.register_generated_candidates(generated_candidates)

            scoring_attempts, scored_candidates = self._attempt_score_mutations(
                generated_candidates
            )
            checkpoint.register_scored_candidates(scored_candidates)

            outcome = self._select_round_outcome(scored_candidates)

            selected_parent_ulid = checkpoint.select_parent_ulid(outcome)
            checkpoint.update_best_from_outcome(outcome)

            round_trace = RoundTrace(
                depth=depth,
                parent_ulid=current_parent_ulid,
                mutation_attempts=mutation_attempts,
                scoring_attempts=scoring_attempts,
                outcome=outcome,
                selected_parent_ulid=selected_parent_ulid,
            )
            checkpoint.append_round_trace(round_trace)

            logger.info(
                "GreedySearch depth={depth} parent_ulid={parent_ulid} "
                "mut_attempts={mut_attempts} mut_ok={mut_ok} "
                "score_attempts={score_attempts} scored={scored} valid={valid} "
                "round_outcome={round_outcome}",
                depth=depth,
                parent_ulid=str(current_parent_ulid),
                mut_attempts=len(mutation_attempts),
                mut_ok=len(generated_candidates),
                score_attempts=len(scoring_attempts),
                scored=len(scored_candidates),
                valid=outcome.num_valid,
                round_outcome=outcome.kind,
            )

            checkpoint.advance_cursor(
                next_depth=depth + 1, parent_ulid=selected_parent_ulid
            )

        return checkpoint

    def resume(self, checkpoint: GreedySearchCheckpoint) -> GreedySearchCheckpoint:
        """Resume a search from a checkpoint (between rounds)."""
        # Continue from checkpoint.cursor.next_depth up to config.max_depth.
        resumed = checkpoint
        # Re-run the same loop by temporarily seeding self.search-style local variables.
        # (Keeps the public API small; search() remains the entrypoint for fresh runs.)
        for depth in range(resumed.cursor.next_depth, self.config.max_depth):
            current_parent_ulid = resumed.cursor.parent_ulid
            parent = resumed.candidates.get(current_parent_ulid)

            mutation_attempts, generated_candidates = self._attempt_generate_mutations(
                parent=parent
            )

            resumed.register_generated_candidates(generated_candidates)

            scoring_attempts, scored_candidates = self._attempt_score_mutations(
                generated_candidates
            )
            resumed.register_scored_candidates(scored_candidates)

            outcome = self._select_round_outcome(scored_candidates)
            selected_parent_ulid = resumed.select_parent_ulid(outcome)
            resumed.update_best_from_outcome(outcome)

            round_trace = RoundTrace(
                depth=depth,
                parent_ulid=current_parent_ulid,
                mutation_attempts=mutation_attempts,
                scoring_attempts=scoring_attempts,
                outcome=outcome,
                selected_parent_ulid=selected_parent_ulid,
            )
            resumed.append_round_trace(round_trace)
            resumed.advance_cursor(
                next_depth=depth + 1, parent_ulid=selected_parent_ulid
            )

        return resumed
