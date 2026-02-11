from unittest.mock import Mock

import pytest

from arid_badger.greedy_search.domain import (
    InvalidEvaluation,
    KernelCandidate,
    ValidEvaluation,
)
from arid_badger.greedy_search.scoring_provider import SerialScoringProvider
from arid_badger.greedy_search.domain import ScoringFailure, ScoringSuccess
from arid_badger.kernelbench.core import KernelScoringResult
from kernelbench.eval import KernelExecResult


def _make_valid_score(*, speedup: float = 2.0) -> KernelScoringResult:
    exec_result = Mock(spec=KernelExecResult)
    exec_result.compiled = True
    exec_result.correctness = True
    exec_result.runtime = 100.0
    exec_result.ref_runtime = 200.0
    exec_result.metadata = {}
    return KernelScoringResult(exec_result=exec_result, speedup=speedup, is_valid=True)


def _make_invalid_score() -> KernelScoringResult:
    exec_result = Mock(spec=KernelExecResult)
    exec_result.compiled = False
    exec_result.correctness = False
    exec_result.runtime = None
    exec_result.ref_runtime = None
    exec_result.metadata = {
        "compilation_error_name": "CompilerError",
        "compilation_error": "syntax error",
    }
    return KernelScoringResult(exec_result=exec_result, speedup=0.0, is_valid=False)


def _make_candidate(code: str = "def kernel(): pass") -> KernelCandidate:
    return KernelCandidate(code=code)


class TestScoreCandidates:
    def test_valid_result(self):
        score = _make_valid_score(speedup=3.0)
        provider = SerialScoringProvider(
            scoring_function=lambda _code, _ref: score,
        )
        candidate = _make_candidate()

        attempts, scored = provider.score_candidates([candidate], "ref_code")

        assert len(attempts) == 1
        assert isinstance(attempts[0], ScoringSuccess)
        assert len(scored) == 1
        assert scored[0][0] is candidate
        assert isinstance(scored[0][1], ValidEvaluation)
        assert scored[0][1].speedup == 3.0

    def test_invalid_result(self):
        score = _make_invalid_score()
        provider = SerialScoringProvider(
            scoring_function=lambda _code, _ref: score,
        )
        candidate = _make_candidate()

        attempts, scored = provider.score_candidates([candidate], "ref_code")

        assert len(attempts) == 1
        assert isinstance(attempts[0], ScoringSuccess)
        assert len(scored) == 1
        assert isinstance(scored[0][1], InvalidEvaluation)
        assert scored[0][1].reason == "compile_failed"

    def test_exception_produces_scoring_failure(self):
        def exploding_scorer(_code: str, _ref: str) -> KernelScoringResult:
            raise RuntimeError("boom")

        provider = SerialScoringProvider(scoring_function=exploding_scorer)
        candidate = _make_candidate()

        attempts, scored = provider.score_candidates([candidate], "ref_code")

        assert len(attempts) == 1
        assert isinstance(attempts[0], ScoringFailure)
        assert "boom" in attempts[0].error.exception_repr
        assert len(scored) == 0

    def test_mixed_batch(self):
        call_count = 0

        def mixed_scorer(_code: str, _ref: str) -> KernelScoringResult:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                raise RuntimeError("second fails")
            return _make_valid_score(speedup=1.5)

        provider = SerialScoringProvider(scoring_function=mixed_scorer)
        candidates = [_make_candidate("a"), _make_candidate("b"), _make_candidate("c")]

        attempts, scored = provider.score_candidates(candidates, "ref_code")

        assert len(attempts) == 3
        assert isinstance(attempts[0], ScoringSuccess)
        assert isinstance(attempts[1], ScoringFailure)
        assert isinstance(attempts[2], ScoringSuccess)
        assert len(scored) == 2


class TestScoreReference:
    def test_valid_reference(self):
        score = _make_valid_score(speedup=1.0)
        provider = SerialScoringProvider(
            scoring_function=lambda _code, _ref: score,
        )

        result = provider.score_reference("class Model:\n    pass")

        assert result.is_ok()
        evaluation = result.unwrap()
        assert isinstance(evaluation, ValidEvaluation)
        assert evaluation.speedup == 1.0

    def test_compile_failed_reference_returns_err(self):
        exec_result = Mock(spec=KernelExecResult)
        exec_result.compiled = False
        exec_result.correctness = False
        exec_result.runtime = None
        exec_result.ref_runtime = None
        exec_result.metadata = {"compilation_error": "fail"}
        score = KernelScoringResult(
            exec_result=exec_result, speedup=0.0, is_valid=False
        )

        provider = SerialScoringProvider(
            scoring_function=lambda _code, _ref: score,
        )

        result = provider.score_reference("class Model:\n    pass")

        assert result.is_err()
        assert "failed to compile" in result.unwrap_err()

    def test_invalid_runtime_returns_err(self):
        exec_result = Mock(spec=KernelExecResult)
        exec_result.compiled = True
        exec_result.correctness = True
        exec_result.runtime = -1.0
        exec_result.ref_runtime = 200.0
        exec_result.metadata = {}
        score = KernelScoringResult(
            exec_result=exec_result, speedup=0.0, is_valid=False
        )

        provider = SerialScoringProvider(
            scoring_function=lambda _code, _ref: score,
        )

        result = provider.score_reference("class Model:\n    pass")

        assert result.is_err()
        assert "invalid runtime" in result.unwrap_err()
