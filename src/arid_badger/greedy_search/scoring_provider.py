from __future__ import annotations

import ast
import math
import traceback
from typing import Callable, List, Sequence, Tuple

from loguru import logger

from arid_badger.kernelbench.core import KernelScoringResult
from arid_badger.typing_utils import implements

from .domain import (
    Evaluation,
    EvaluationMetrics,
    InvalidEvaluation,
    KernelCandidate,
    ScoringAttempt,
    ScoringError,
    ScoringFailure,
    ScoringProvider,
    ScoringSuccess,
    ValidEvaluation,
    execution_feedback_from_exec_result,
)


def _has_class_def(source: str, name: str) -> bool:
    """Check if a class definition with the given name exists in the source code."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return False
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == name:
            return True
    return False


def _ensure_modelnew_entry_point(source: str) -> str:
    """Ensure kernel code has a ModelNew class for KernelBench compatibility."""
    if _has_class_def(source, "ModelNew"):
        return source
    return f"{source.rstrip()}\n\nclass ModelNew(Model):\n    pass\n"


def _score_to_evaluation(score: KernelScoringResult) -> Evaluation:
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


def _validate_reference_score(score: KernelScoringResult) -> None:
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


def _short_ulid(ulid: object | None) -> str:
    if ulid is None:
        return "none"
    return str(ulid)[:6]


@implements(ScoringProvider)
class SerialScoringProvider:
    """Scores kernel candidates serially using a provided scoring function."""

    def __init__(
        self,
        scoring_function: Callable[[str, str], KernelScoringResult],
    ) -> None:
        self._scoring_function = scoring_function

    def score_candidates(
        self,
        candidates: Sequence[KernelCandidate],
        reference_kernel_code: str,
    ) -> tuple[List[ScoringAttempt], List[Tuple[KernelCandidate, Evaluation]]]:
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
                raw_score = self._scoring_function(
                    candidate.code, reference_kernel_code
                )
                evaluation = _score_to_evaluation(raw_score)
                attempts.append(
                    ScoringSuccess(
                        candidate_ulid=candidate.ulid, evaluation=evaluation
                    )
                )
                scored.append((candidate, evaluation))
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
                    "Scoring failed candidate_ulid={candidate_ulid} error={error}",
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

    def score_reference(self, reference_kernel_code: str) -> Evaluation:
        custom_baseline_code = _ensure_modelnew_entry_point(reference_kernel_code)
        score = self._scoring_function(custom_baseline_code, reference_kernel_code)
        _validate_reference_score(score)
        return _score_to_evaluation(score)
