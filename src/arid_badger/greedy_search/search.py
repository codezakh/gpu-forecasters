from __future__ import annotations

import traceback
from collections import Counter
from typing import Callable, Dict, List, Literal, Optional, Tuple

from arid_badger.typing_utils import Option, is_ok
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


def _summarize_attempt_failures(
    *,
    mutation_attempts: List[MutationAttempt],
    scoring_attempts: List[ScoringAttempt],
) -> Dict[str, int]:
    mutation_failures = sum(
        1 for attempt in mutation_attempts if isinstance(attempt, MutationFailure)
    )
    scoring_failures = sum(
        1 for attempt in scoring_attempts if isinstance(attempt, ScoringFailure)
    )
    return {
        "mutation_failures": mutation_failures,
        "scoring_failures": scoring_failures,
    }


def _summarize_invalid_reasons(
    scored_candidates: List[Tuple[KernelCandidate, Evaluation]],
) -> Dict[str, int]:
    reasons = Counter()
    for _, evaluation in scored_candidates:
        if isinstance(evaluation, InvalidEvaluation):
            reasons[evaluation.reason] += 1
    return dict(reasons)


def _log_run_start(
    *, config: "GreedySearchConfig", checkpoint: GreedySearchCheckpoint
) -> None:
    best_kind = checkpoint.best_evaluation.kind
    best_speedup: Optional[float] = None
    if isinstance(checkpoint.best_evaluation, ValidEvaluation):
        best_speedup = checkpoint.best_evaluation.speedup

    logger.info(
        "GreedySearch start max_depth={max_depth} next_depth={next_depth} "
        "parent_ulid={parent_ulid} best_kind={best_kind} best_speedup={best_speedup}",
        max_depth=config.max_depth,
        next_depth=checkpoint.cursor.next_depth,
        parent_ulid=str(checkpoint.cursor.parent_ulid),
        best_kind=best_kind,
        best_speedup=best_speedup,
    )


def _log_round_outcome_and_policy(
    *,
    depth: int,
    current_parent: KernelCandidate,
    selected_parent: KernelCandidate,
    outcome: RoundOutcome,
    incumbent_best: Evaluation,
    best_updated: bool,
    mutation_attempts: List[MutationAttempt],
    scoring_attempts: List[ScoringAttempt],
    scored_candidates: List[Tuple[KernelCandidate, Evaluation]],
) -> None:
    failure_summary = _summarize_attempt_failures(
        mutation_attempts=mutation_attempts, scoring_attempts=scoring_attempts
    )
    invalid_reasons = _summarize_invalid_reasons(scored_candidates)
    parent_changed = current_parent.ulid != selected_parent.ulid

    winner_ulid: Optional[str] = None
    winner_speedup: Optional[float] = None
    if isinstance(outcome, RoundWinnerSelected):
        winner_ulid = str(outcome.winner_ulid)
        winner_speedup = outcome.winner_speedup

    incumbent_kind = incumbent_best.kind
    incumbent_speedup: Optional[float] = None
    if isinstance(incumbent_best, ValidEvaluation):
        incumbent_speedup = incumbent_best.speedup

    logger.info(
        "GreedySearch round depth={depth} parent_ulid={parent_ulid} "
        "outcome={outcome} selected_parent_ulid={selected_parent_ulid} "
        "parent_changed={parent_changed} best_updated={best_updated} "
        "num_scored={num_scored} num_valid={num_valid} "
        "winner_ulid={winner_ulid} winner_speedup={winner_speedup} "
        "incumbent_kind={incumbent_kind} incumbent_speedup={incumbent_speedup} "
        "mutation_failures={mutation_failures} scoring_failures={scoring_failures} "
        "invalid_reasons={invalid_reasons}",
        depth=depth,
        parent_ulid=str(current_parent.ulid),
        outcome=outcome.kind,
        selected_parent_ulid=str(selected_parent.ulid),
        parent_changed=parent_changed,
        best_updated=best_updated,
        num_scored=outcome.num_scored,
        num_valid=outcome.num_valid,
        winner_ulid=winner_ulid,
        winner_speedup=winner_speedup,
        incumbent_kind=incumbent_kind,
        incumbent_speedup=incumbent_speedup,
        mutation_failures=failure_summary["mutation_failures"],
        scoring_failures=failure_summary["scoring_failures"],
        invalid_reasons=invalid_reasons,
    )


def _select_next_parent(
    *,
    current_parent: KernelCandidate,
    outcome: RoundOutcome,
    candidates: CandidateGraph,
) -> KernelCandidate:
    """Greedy policy: follow the round winner if available; otherwise keep parent.

    Note: The outcome is ULID-referential for trace/checkpoint stability; we
    resolve that to an entity here so the search loop can operate on candidates.
    """
    if isinstance(outcome, RoundWinnerSelected):
        selected = candidates.get(outcome.winner_ulid)
        logger.info(
            "GreedySearch decision next_parent=winner "
            "current_parent_ulid={current_parent_ulid} winner_ulid={winner_ulid}",
            current_parent_ulid=str(current_parent.ulid),
            winner_ulid=str(outcome.winner_ulid),
        )
        return selected
    logger.info(
        "GreedySearch decision next_parent=incumbent "
        "current_parent_ulid={current_parent_ulid} outcome={outcome}",
        current_parent_ulid=str(current_parent.ulid),
        outcome=outcome.kind,
    )
    return current_parent


def _select_best_update_from_outcome(
    *,
    incumbent_best: Evaluation,
    outcome: RoundOutcome,
    candidates: CandidateGraph,
) -> Option[KernelCandidate, Literal["no_update"]]:
    """Greedy policy: update best-so-far if this round produced a strictly better valid winner.

    "Best so far" is not necessarily "best valid" — it's the best candidate we've *seen* so far.
    Concretely:
    - Any valid winner beats a currently-invalid incumbent.
    - If both are valid, higher speedup wins (ties do not replace incumbent).
    """
    if not isinstance(outcome, RoundWinnerSelected):
        logger.info(
            "GreedySearch decision best_update=skip reason=no_winner outcome={outcome}",
            outcome=outcome.kind,
        )
        return Option.err("no_update")

    winner = candidates.get(outcome.winner_ulid)

    # Any valid winner beats a currently-invalid incumbent.
    if isinstance(incumbent_best, InvalidEvaluation):
        logger.info(
            "GreedySearch decision best_update=winner reason=incumbent_invalid "
            "winner_ulid={winner_ulid}",
            winner_ulid=str(outcome.winner_ulid),
        )
        return Option.ok(winner)

    # If both are valid, prefer higher speedup.
    if isinstance(incumbent_best, ValidEvaluation):
        if outcome.winner_speedup > incumbent_best.speedup:
            logger.info(
                "GreedySearch decision best_update=winner reason=speedup_improved "
                "winner_ulid={winner_ulid} winner_speedup={winner_speedup} "
                "incumbent_speedup={incumbent_speedup}",
                winner_ulid=str(outcome.winner_ulid),
                winner_speedup=outcome.winner_speedup,
                incumbent_speedup=incumbent_best.speedup,
            )
            return Option.ok(winner)
        logger.info(
            "GreedySearch decision best_update=skip reason=no_improvement "
            "winner_ulid={winner_ulid} winner_speedup={winner_speedup} "
            "incumbent_speedup={incumbent_speedup}",
            winner_ulid=str(outcome.winner_ulid),
            winner_speedup=outcome.winner_speedup,
            incumbent_speedup=incumbent_best.speedup,
        )

    return Option.err("no_update")


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

    def _create_initial_checkpoint(self) -> GreedySearchCheckpoint:
        """Create the initial checkpoint for a fresh run (includes starter scoring)."""
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
        checkpoint.validate_invariants()
        return checkpoint

    def _score_to_evaluation(self, score: KernelScoringResult) -> Evaluation:
        exec_result = score.exec_result
        metrics = EvaluationMetrics(
            compiled=exec_result.compiled,
            correctness=exec_result.correctness,
            runtime=exec_result.runtime,
            ref_runtime=exec_result.ref_runtime,
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
                logger.warning(
                    "GreedySearch mutation_failed attempt_idx={attempt_idx} "
                    "parent_ulid={parent_ulid} error={error}",
                    attempt_idx=attempt_idx,
                    parent_ulid=str(parent.ulid),
                    error=repr(e),
                )
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
                logger.warning(
                    "GreedySearch scoring_failed candidate_ulid={candidate_ulid} "
                    "error={error}",
                    candidate_ulid=str(candidate.ulid),
                    error=repr(e),
                )
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
            logger.info(
                "GreedySearch decision round_outcome=no_evaluations num_scored=0"
            )
            return RoundNoEvaluations()

        valid: List[Tuple[KernelCandidate, ValidEvaluation]] = []
        for candidate, evaluation in scored_candidates:
            if isinstance(evaluation, ValidEvaluation):
                valid.append((candidate, evaluation))

        if not valid:
            invalid_reasons = _summarize_invalid_reasons(scored_candidates)
            logger.info(
                "GreedySearch decision round_outcome=all_evaluations_invalid "
                "num_scored={num_scored} invalid_reasons={invalid_reasons}",
                num_scored=len(scored_candidates),
                invalid_reasons=invalid_reasons,
            )
            return RoundAllEvaluationsInvalid(num_scored=len(scored_candidates))

        best_candidate, best_evaluation = max(valid, key=lambda ce: ce[1].speedup)
        logger.info(
            "GreedySearch decision round_outcome=winner_selected "
            "winner_ulid={winner_ulid} winner_speedup={winner_speedup} "
            "num_scored={num_scored} num_valid={num_valid}",
            winner_ulid=str(best_candidate.ulid),
            winner_speedup=best_evaluation.speedup,
            num_scored=len(scored_candidates),
            num_valid=len(valid),
        )
        return RoundWinnerSelected(
            winner_ulid=best_candidate.ulid,
            winner_evaluation=best_evaluation,
            num_scored=len(scored_candidates),
            num_valid=len(valid),
        )

    def _run_from_checkpoint(
        self, checkpoint: GreedySearchCheckpoint
    ) -> GreedySearchCheckpoint:
        """Advance a run from the given checkpoint to config.max_depth.

        Mutates and returns the same checkpoint object (in-place).
        """
        checkpoint.validate_invariants()

        for depth in range(checkpoint.cursor.next_depth, self.config.max_depth):
            current_parent = checkpoint.current_parent()

            mutation_attempts, generated_candidates = self._attempt_generate_mutations(
                parent=current_parent
            )

            # Add generated candidates to the graph immediately so we can reference them by ULID.
            checkpoint.register_generated_candidates(generated_candidates)

            scoring_attempts, scored_candidates = self._attempt_score_mutations(
                generated_candidates
            )
            checkpoint.register_scored_candidates(scored_candidates)

            outcome = self._select_round_outcome(scored_candidates)

            selected_parent = _select_next_parent(
                current_parent=current_parent,
                outcome=outcome,
                candidates=checkpoint.candidates,
            )
            incumbent_best = checkpoint.best_evaluation
            best_update_ulid = _select_best_update_from_outcome(
                incumbent_best=incumbent_best,
                outcome=outcome,
                candidates=checkpoint.candidates,
            )
            best_updated = is_ok(best_update_ulid)
            if is_ok(best_update_ulid):
                checkpoint.set_best_candidate(candidate=best_update_ulid.unwrap())

            round_trace = RoundTrace(
                depth=depth,
                parent_ulid=current_parent.ulid,
                mutation_attempts=mutation_attempts,
                scoring_attempts=scoring_attempts,
                outcome=outcome,
                selected_parent_ulid=selected_parent.ulid,
            )
            checkpoint.append_round_trace(round_trace)

            _log_round_outcome_and_policy(
                depth=depth,
                current_parent=current_parent,
                selected_parent=selected_parent,
                outcome=outcome,
                incumbent_best=incumbent_best,
                best_updated=best_updated,
                mutation_attempts=mutation_attempts,
                scoring_attempts=scoring_attempts,
                scored_candidates=scored_candidates,
            )

            checkpoint.advance_parent(next_depth=depth + 1, parent=selected_parent)

        return checkpoint

    def run(
        self, *, checkpoint: Optional[GreedySearchCheckpoint] = None
    ) -> GreedySearchCheckpoint:
        """Run greedy search either from scratch or from an existing checkpoint."""
        if checkpoint is None:
            checkpoint = self._create_initial_checkpoint()
        _log_run_start(config=self.config, checkpoint=checkpoint)
        return self._run_from_checkpoint(checkpoint)

    def search(self) -> GreedySearchCheckpoint:
        """Perform greedy search for optimized kernels (fresh run)."""
        return self.run()

    def resume(self, checkpoint: GreedySearchCheckpoint) -> GreedySearchCheckpoint:
        """Resume a search from a checkpoint (between rounds)."""
        return self.run(checkpoint=checkpoint)
