from __future__ import annotations

import ast
import math
import traceback
from collections import Counter
from typing import Callable, Dict, List, Literal, Optional, Tuple

import attrs
from loguru import logger

from arid_badger.kernelbench.core import KernelScoringResult
from arid_badger.typing_utils import Option, is_ok

from .checkpoint import GreedySearchCheckpoint, SearchCursor
from .domain import (
    CandidateGraph,
    Evaluation,
    EvaluationMetrics,
    KernelCandidate,
    MutationContext,
    MutationFunction,
    execution_feedback_from_exec_result,
)
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


def _short_ulid(ulid: Optional[object]) -> str:
    if ulid is None:
        return "none"
    text = str(ulid)
    return text[:6]


def _has_class_def(source: str, name: str) -> bool:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return True
    return False


def _ensure_modelnew_entry_point(source: str) -> str:
    if _has_class_def(source, "ModelNew"):
        return source
    return f"{source.rstrip()}\n\nclass ModelNew(Model):\n    pass\n"


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
        "Starting greedy search at depth {next_depth}/{max_depth} (parent={parent_short})",
        next_depth=checkpoint.cursor.next_depth,
        max_depth=config.max_depth,
        parent_short=_short_ulid(checkpoint.cursor.parent_ulid),
    )
    logger.debug(
        "GreedySearch start details max_depth={max_depth} next_depth={next_depth} "
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

    if isinstance(outcome, RoundWinnerSelected):
        logger.success(
            "Round {depth} finished: winner selected (winner={winner_short}, speedup={winner_speedup:.4f}x)",
            depth=depth,
            winner_short=_short_ulid(outcome.winner_ulid),
            winner_speedup=outcome.winner_speedup,
        )
    else:
        logger.warning(
            "Round {depth} finished: no winner (outcome={outcome})",
            depth=depth,
            outcome=outcome.kind,
        )
    if parent_changed:
        logger.info(
            "Round {depth} event: parent updated for next round.",
            depth=depth,
        )
    else:
        logger.info(
            "Round {depth} event: parent retained for next round.",
            depth=depth,
        )
    if best_updated:
        logger.success(
            "Round {depth} event: best-so-far updated.",
            depth=depth,
        )
    else:
        logger.info(
            "Round {depth} event: best-so-far unchanged.",
            depth=depth,
        )
    logger.debug(
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
            "Next parent: use round winner (winner={winner_short}, previous_parent={parent_short})",
            winner_short=_short_ulid(outcome.winner_ulid),
            parent_short=_short_ulid(current_parent.ulid),
        )
        logger.debug(
            "GreedySearch next_parent details current_parent_ulid={current_parent_ulid} winner_ulid={winner_ulid}",
            current_parent_ulid=str(current_parent.ulid),
            winner_ulid=str(outcome.winner_ulid),
        )
        return selected
    logger.info(
        "Next parent: keep current parent (no winner this round).",
    )
    logger.debug(
        "GreedySearch next_parent details current_parent_ulid={current_parent_ulid} outcome={outcome}",
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
            "Best-so-far unchanged: no winner this round.",
        )
        logger.debug(
            "GreedySearch best_update details outcome={outcome}",
            outcome=outcome.kind,
        )
        return Option.err("no_update")

    winner = candidates.get(outcome.winner_ulid)

    # Any valid winner beats a currently-invalid incumbent.
    if isinstance(incumbent_best, InvalidEvaluation):
        logger.success(
            "Best-so-far updated: previous best invalid, winner becomes new best (winner={winner_short}).",
            winner_short=_short_ulid(outcome.winner_ulid),
        )
        logger.debug(
            "GreedySearch best_update details reason=incumbent_invalid winner_ulid={winner_ulid}",
            winner_ulid=str(outcome.winner_ulid),
        )
        return Option.ok(winner)

    # If both are valid, prefer higher speedup.
    if isinstance(incumbent_best, ValidEvaluation):
        if outcome.winner_speedup > incumbent_best.speedup:
            logger.success(
                "Best-so-far updated: winner speedup improved "
                "(winner={winner_short}, speedup={winner_speedup:.4f}x).",
                winner_short=_short_ulid(outcome.winner_ulid),
                winner_speedup=outcome.winner_speedup,
            )
            logger.debug(
                "GreedySearch best_update details reason=speedup_improved "
                "winner_ulid={winner_ulid} winner_speedup={winner_speedup} "
                "incumbent_speedup={incumbent_speedup}",
                winner_ulid=str(outcome.winner_ulid),
                winner_speedup=outcome.winner_speedup,
                incumbent_speedup=incumbent_best.speedup,
            )
            return Option.ok(winner)
        logger.info(
            "Best-so-far unchanged: winner did not improve speedup "
            "(winner={winner_short}, speedup={winner_speedup:.4f}x).",
            winner_short=_short_ulid(outcome.winner_ulid),
            winner_speedup=outcome.winner_speedup,
        )
        logger.debug(
            "GreedySearch best_update details reason=no_improvement "
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


class GreedySearch:
    def __init__(self, config: GreedySearchConfig):
        """
        Initialize a GreedySearch instance.

        Args:
            config: Search configuration containing all search parameters and dependencies
        """
        self.config = config

    def _validate_reference_score(self, score: KernelScoringResult) -> None:
        exec_result = score.exec_result
        metadata = exec_result.metadata or {}
        if exec_result.compiled is not True:
            raise ValueError(
                "Reference kernel failed to compile. " f"metadata={metadata!r}"
            )
        if exec_result.correctness is not True:
            raise ValueError(
                "Reference kernel failed correctness checks. " f"metadata={metadata!r}"
            )
        if (
            exec_result.runtime is None
            or not math.isfinite(exec_result.runtime)
            or exec_result.runtime <= 0
        ):
            raise ValueError(
                "Reference kernel produced invalid runtime. "
                f"runtime={exec_result.runtime!r} metadata={metadata!r}"
            )
        if (
            exec_result.ref_runtime is None
            or not math.isfinite(exec_result.ref_runtime)
            or exec_result.ref_runtime <= 0
        ):
            raise ValueError(
                "Reference kernel produced invalid ref runtime. "
                f"ref_runtime={exec_result.ref_runtime!r} metadata={metadata!r}"
            )

    def score_reference_kernel_only(self) -> KernelScoringResult:
        custom_baseline_code = _ensure_modelnew_entry_point(
            self.config.reference_kernel_code
        )
        score = self.config.scoring_function(
            custom_baseline_code, self.config.reference_kernel_code
        )
        self._validate_reference_score(score)
        return score

    def _create_initial_checkpoint(self) -> GreedySearchCheckpoint:
        """Create the initial checkpoint for a fresh run (includes starter scoring)."""
        starter_code = _ensure_modelnew_entry_point(self.config.starter_kernel_code)
        starter_raw_score = self.config.scoring_function(
            starter_code, self.config.reference_kernel_code
        )
        starter_evaluation = self._score_to_evaluation(starter_raw_score)

        starter_candidate = KernelCandidate(
            code=starter_code,
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
        execution_feedback = execution_feedback_from_exec_result(
            exec_result=exec_result, speedup=score.speedup, is_valid=score.is_valid
        )

        if score.is_valid:
            return ValidEvaluation(
                speedup=score.speedup,
                metrics=metrics,
                execution_feedback=execution_feedback,
            )

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
        return InvalidEvaluation(
            reason=reason,
            metrics=metrics,
            execution_feedback=execution_feedback,
        )

    def _attempt_generate_mutations(
        self, *, parent: KernelCandidate
    ) -> Tuple[List[MutationAttempt], List[KernelCandidate]]:
        mutation_context = MutationContext(
            reference_kernel_code=self.config.reference_kernel_code,
            previous_kernel_code=parent.code,
            previous_kernel_ulid=parent.ulid,
            previous_evaluation=parent.evaluation,
            backend=self.config.backend,
            precision=self.config.precision,
        )

        attempts: List[MutationAttempt] = []
        generated: List[KernelCandidate] = []
        logger.info(
            "Generating {num_mutations} candidate(s) from parent {parent_short}",
            num_mutations=self.config.num_mutations,
            parent_short=_short_ulid(parent.ulid),
        )
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
                logger.success(
                    "Generated candidate {attempt_idx} ulid={candidate_short}",
                    attempt_idx=attempt_idx,
                    candidate_short=_short_ulid(candidate.ulid),
                )
                generated.append(candidate)
            except Exception as e:
                logger.error(
                    "GreedySearch mutation_failed attempt_idx={attempt_idx} "
                    "parent_ulid={parent_ulid} error={error}",
                    attempt_idx=attempt_idx,
                    parent_ulid=str(parent.ulid),
                    error=repr(e),
                )
                logger.debug(
                    "Mutation failure traceback:\n{traceback}",
                    traceback=traceback.format_exc(),
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
        for idx, candidate in enumerate(candidates, start=1):
            try:
                logger.info(
                    "Scoring candidate {idx}/{total} (compiling + benchmarking) ulid={candidate_short}",
                    idx=idx,
                    total=len(candidates),
                    candidate_short=_short_ulid(candidate.ulid),
                )
                raw_score = self.config.scoring_function(
                    candidate.code, self.config.reference_kernel_code
                )
                evaluation = self._score_to_evaluation(raw_score)
                pair = (candidate, evaluation)
                attempts.append(
                    ScoringSuccess(candidate_ulid=candidate.ulid, evaluation=evaluation)
                )
                scored.append(pair)
                if isinstance(evaluation, ValidEvaluation):
                    logger.success(
                        "Scored candidate ulid={candidate_short} speedup={speedup:.4f}x",
                        candidate_short=_short_ulid(candidate.ulid),
                        speedup=evaluation.speedup,
                    )
                else:
                    logger.warning(
                        "Scored candidate ulid={candidate_short} invalid_reason={reason}",
                        candidate_short=_short_ulid(candidate.ulid),
                        reason=evaluation.reason,
                    )
            except Exception as e:
                logger.error(
                    "GreedySearch scoring_failed candidate_ulid={candidate_ulid} "
                    "error={error}",
                    candidate_ulid=str(candidate.ulid),
                    error=repr(e),
                )
                logger.debug(
                    "Scoring failure traceback:\n{traceback}",
                    traceback=traceback.format_exc(),
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
            logger.warning("Round event: no candidates were scored.")
            logger.debug(
                "GreedySearch decision round_outcome=no_evaluations num_scored=0"
            )
            return RoundNoEvaluations()

        valid: List[Tuple[KernelCandidate, ValidEvaluation]] = []
        for candidate, evaluation in scored_candidates:
            if isinstance(evaluation, ValidEvaluation):
                valid.append((candidate, evaluation))

        if not valid:
            invalid_reasons = _summarize_invalid_reasons(scored_candidates)
            logger.warning(
                "Round event: all candidates invalid (num_scored={num_scored}).",
                num_scored=len(scored_candidates),
            )
            logger.debug(
                "GreedySearch decision round_outcome=all_evaluations_invalid "
                "num_scored={num_scored} invalid_reasons={invalid_reasons}",
                num_scored=len(scored_candidates),
                invalid_reasons=invalid_reasons,
            )
            return RoundAllEvaluationsInvalid(num_scored=len(scored_candidates))

        best_candidate, best_evaluation = max(valid, key=lambda ce: ce[1].speedup)
        logger.success(
            "Round event: winner selected (winner={winner_short}, speedup={winner_speedup:.4f}x).",
            winner_short=_short_ulid(best_candidate.ulid),
            winner_speedup=best_evaluation.speedup,
        )
        logger.debug(
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

    def is_complete(self, checkpoint: GreedySearchCheckpoint) -> bool:
        """Return True if the search has finished all planned rounds."""
        return checkpoint.cursor.next_depth >= self.config.max_depth

    def _run_from_checkpoint_until(
        self, checkpoint: GreedySearchCheckpoint, *, stop_depth_exclusive: int
    ) -> GreedySearchCheckpoint:
        """Advance a run from the given checkpoint up to stop_depth_exclusive.

        Mutates and returns the same checkpoint object (in-place).
        """
        checkpoint.validate_invariants()
        if stop_depth_exclusive <= checkpoint.cursor.next_depth:
            return checkpoint

        for depth in range(checkpoint.cursor.next_depth, stop_depth_exclusive):
            current_parent = checkpoint.current_parent()
            logger.info(
                "Round {depth} starting (parent {parent_short})",
                depth=depth,
                parent_short=_short_ulid(current_parent.ulid),
            )

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
        return self._run_from_checkpoint_until(
            checkpoint, stop_depth_exclusive=self.config.max_depth
        )

    def search(self) -> GreedySearchCheckpoint:
        """Perform greedy search for optimized kernels (fresh run)."""
        return self.run()

    def resume(self, checkpoint: GreedySearchCheckpoint) -> GreedySearchCheckpoint:
        """Resume a search from a checkpoint (between rounds)."""
        return self.run(checkpoint=checkpoint)

    def step(self, checkpoint: GreedySearchCheckpoint) -> GreedySearchCheckpoint:
        """Advance a search by at most one round (no-op if complete)."""
        next_stop = min(checkpoint.cursor.next_depth + 1, self.config.max_depth)
        return self._run_from_checkpoint_until(
            checkpoint, stop_depth_exclusive=next_stop
        )
