from unittest.mock import MagicMock, Mock

import pytest

from arid_badger.greedy_search.domain import (
    MutationContext,
    MutationFunction,
    MutatedKernel,
)
from arid_badger.greedy_search.domain import ValidEvaluation
from arid_badger.greedy_search.trace import MutationFailure, ScoringFailure
from arid_badger.greedy_search.search import (
    GreedySearch,
    GreedySearchConfig,
)
from arid_badger.kernelbench.core import KernelScoringResult
from kernelbench.eval import KernelExecResult


@pytest.fixture
def mock_mutation_function():
    """Create a mock MutationFunction that returns predictable MutatedKernel objects."""
    mock_fn = MagicMock(spec=MutationFunction)
    return mock_fn


@pytest.fixture
def mock_scoring_function():
    """Create a mock scoring function that returns predictable KernelScoringResult objects."""

    def _scoring_fn(mutated_code: str, reference_code: str) -> KernelScoringResult:
        # Create a mock exec_result
        exec_result = Mock(spec=KernelExecResult)
        exec_result.compiled = True
        exec_result.correctness = True
        exec_result.runtime = 100.0
        exec_result.ref_runtime = 200.0
        exec_result.metadata = {}

        # Determine speedup based on kernel code (for testing different scenarios)
        # If code contains "fast", give it high speedup
        # If code contains "slow", give it low speedup
        # If code contains "invalid", mark as invalid
        if "invalid" in mutated_code:
            exec_result.correctness = False
            result = KernelScoringResult(
                exec_result=exec_result, speedup=0.0, is_valid=False
            )
        elif "fast" in mutated_code:
            speedup = 3.0
            result = KernelScoringResult(
                exec_result=exec_result, speedup=speedup, is_valid=True
            )
        elif "slow" in mutated_code:
            speedup = 0.5
            result = KernelScoringResult(
                exec_result=exec_result, speedup=speedup, is_valid=True
            )
        else:
            speedup = 1.0
            result = KernelScoringResult(
                exec_result=exec_result, speedup=speedup, is_valid=True
            )

        return result

    return _scoring_fn


@pytest.fixture
def starter_kernel_code():
    """Sample starter kernel code."""
    return "def kernel(x): return x * 2"


@pytest.fixture
def reference_kernel_code():
    """Sample reference kernel code."""
    return "def kernel(x): return x * 2"


def test_basic_search_single_depth_two_mutations(
    mock_mutation_function,
    mock_scoring_function,
    starter_kernel_code,
    reference_kernel_code,
):
    """Test basic search with 1 depth level and 2 mutations."""
    # Setup: mock mutation function returns two mutations
    mutation1 = MutatedKernel(
        kernel_code="def kernel(x): return x * 2  # fast",
        ancestor_ulid=None,
    )
    mutation2 = MutatedKernel(
        kernel_code="def kernel(x): return x * 2  # slow",
        ancestor_ulid=None,
    )
    mock_mutation_function.side_effect = [mutation1, mutation2]

    # Create search config
    config = GreedySearchConfig(
        max_depth=1,
        num_mutations=2,
        starter_kernel_code=starter_kernel_code,
        reference_kernel_code=reference_kernel_code,
        mutation_function=mock_mutation_function,
        scoring_function=mock_scoring_function,
    )
    search = GreedySearch(config=config)

    # Run search
    result = search.run()

    # Verify best kernel is the one with highest speedup (mutation1 with "fast")
    assert len(result.rounds) == 1
    assert len(result.evaluated_candidates()) == 3  # starter + 2 scored mutations
    assert result.rounds[0].outcome.kind == "winner_selected"
    assert "fast" in result.best_candidate().code
    assert isinstance(result.best_evaluation, ValidEvaluation)
    assert result.best_evaluation.speedup == 3.0


def test_search_multiple_depth_levels(
    mock_mutation_function,
    mock_scoring_function,
    starter_kernel_code,
    reference_kernel_code,
):
    """Test search with multiple depth levels."""
    # Setup: mock mutation function returns different mutations at each depth
    # Depth 0: mutation1 (fast) and mutation2 (slow) -> pick mutation1
    # Depth 1: mutation3 (fast) and mutation4 (fast) -> pick mutation3
    mutation1 = MutatedKernel(
        kernel_code="def kernel(x): return x * 2  # fast",
        ancestor_ulid=None,
    )
    mutation2 = MutatedKernel(
        kernel_code="def kernel(x): return x * 2  # slow",
        ancestor_ulid=None,
    )
    mutation3 = MutatedKernel(
        kernel_code="def kernel(x): return x * 3  # fast",
        ancestor_ulid=mutation1.ulid,
    )
    mutation4 = MutatedKernel(
        kernel_code="def kernel(x): return x * 4  # fast",
        ancestor_ulid=mutation1.ulid,
    )

    mock_mutation_function.side_effect = [
        mutation1,
        mutation2,  # Depth 0
        mutation3,
        mutation4,  # Depth 1
    ]

    config = GreedySearchConfig(
        max_depth=2,
        num_mutations=2,
        starter_kernel_code=starter_kernel_code,
        reference_kernel_code=reference_kernel_code,
        mutation_function=mock_mutation_function,
        scoring_function=mock_scoring_function,
    )
    search = GreedySearch(config=config)

    result = search.run()

    # Verify search history contains all evaluated kernels
    # Starter + 2 mutations at depth 0 + 2 mutations at depth 1 = 5 total
    assert len(result.rounds) == 2
    assert len(result.evaluated_candidates()) == 5

    # Verify best kernel is one of the fast mutations
    assert isinstance(result.best_evaluation, ValidEvaluation)
    assert result.best_evaluation.speedup == 3.0
    assert result.rounds[0].outcome.kind == "winner_selected"
    assert result.rounds[1].outcome.kind == "winner_selected"
    round1_parent = result.candidates.get(result.rounds[1].parent_ulid)
    assert "fast" in round1_parent.code


def test_search_best_kernel_improves_over_rounds(
    mock_mutation_function,
    mock_scoring_function,
    starter_kernel_code,
    reference_kernel_code,
):
    """Test search where best kernel improves over rounds."""
    # Setup: mutations get progressively better
    mutation1 = MutatedKernel(
        kernel_code="def kernel(x): return x * 2  # slow",  # speedup 0.5
        ancestor_ulid=None,
    )
    mutation2 = MutatedKernel(
        kernel_code="def kernel(x): return x * 2",  # speedup 1.0
        ancestor_ulid=None,
    )
    mutation3 = MutatedKernel(
        kernel_code="def kernel(x): return x * 2  # fast",  # speedup 3.0
        ancestor_ulid=mutation2.ulid,
    )
    mutation4 = MutatedKernel(
        kernel_code="def kernel(x): return x * 2  # slow",  # speedup 0.5
        ancestor_ulid=mutation2.ulid,
    )

    mock_mutation_function.side_effect = [
        mutation1,
        mutation2,  # Round 0: best is mutation2 (speedup 1.0)
        mutation3,
        mutation4,  # Round 1: best is mutation3 (speedup 3.0)
    ]

    config = GreedySearchConfig(
        max_depth=2,
        num_mutations=2,
        starter_kernel_code=starter_kernel_code,
        reference_kernel_code=reference_kernel_code,
        mutation_function=mock_mutation_function,
        scoring_function=mock_scoring_function,
    )
    search = GreedySearch(config=config)

    result = search.run()

    # Verify overall best is mutation3 (highest speedup across all rounds)
    assert len(result.rounds) == 2
    assert isinstance(result.best_evaluation, ValidEvaluation)
    assert result.best_evaluation.speedup == 3.0
    assert "fast" in result.best_candidate().code


def test_search_no_valid_mutations_continues_with_same_parent(
    mock_mutation_function,
    mock_scoring_function,
    starter_kernel_code,
    reference_kernel_code,
):
    """Test search where no valid mutations are found (continues with same parent)."""
    # Setup: all mutations are invalid across all rounds
    mutation1 = MutatedKernel(
        kernel_code="def kernel(x): return x * 2  # invalid",
        ancestor_ulid=None,
    )
    mutation2 = MutatedKernel(
        kernel_code="def kernel(x): return x * 2  # invalid",
        ancestor_ulid=None,
    )

    # Use distinct objects per call so candidates have distinct ULIDs.
    mock_mutation_function.side_effect = [
        mutation1,
        mutation2,
        MutatedKernel(kernel_code=mutation1.kernel_code, ancestor_ulid=None),
        MutatedKernel(kernel_code=mutation2.kernel_code, ancestor_ulid=None),
        MutatedKernel(kernel_code=mutation1.kernel_code, ancestor_ulid=None),
        MutatedKernel(kernel_code=mutation2.kernel_code, ancestor_ulid=None),
    ]

    config = GreedySearchConfig(
        max_depth=3,
        num_mutations=2,
        starter_kernel_code=starter_kernel_code,
        reference_kernel_code=reference_kernel_code,
        mutation_function=mock_mutation_function,
        scoring_function=mock_scoring_function,
    )
    search = GreedySearch(config=config)

    result = search.run()

    # Best kernel should be the starter (since no valid mutations found)
    assert result.best_candidate().code == starter_kernel_code
    assert len(result.rounds) == 3
    assert all(r.outcome.kind == "all_evaluations_invalid" for r in result.rounds)
    starter_ulid = result.rounds[0].parent_ulid
    assert all(r.selected_parent_ulid == starter_ulid for r in result.rounds)
    assert (
        len(result.evaluated_candidates()) == 7
    )  # starter + (2 invalid scored) * 3 rounds


def test_search_all_mutations_invalid(
    mock_mutation_function,
    mock_scoring_function,
    starter_kernel_code,
    reference_kernel_code,
):
    """Test search where all mutations are invalid."""
    mutation1 = MutatedKernel(
        kernel_code="def kernel(x): return x * 2  # invalid",
        ancestor_ulid=None,
    )
    mutation2 = MutatedKernel(
        kernel_code="def kernel(x): return x * 2  # invalid",
        ancestor_ulid=None,
    )

    mock_mutation_function.side_effect = [mutation1, mutation2]

    config = GreedySearchConfig(
        max_depth=1,
        num_mutations=2,
        starter_kernel_code=starter_kernel_code,
        reference_kernel_code=reference_kernel_code,
        mutation_function=mock_mutation_function,
        scoring_function=mock_scoring_function,
    )
    search = GreedySearch(config=config)

    result = search.run()

    # Best kernel should be starter
    assert len(result.rounds) == 1
    assert result.rounds[0].outcome.kind == "all_evaluations_invalid"
    assert result.best_candidate().code == starter_kernel_code
    # All mutations should be evaluated (even if invalid)
    assert len(result.evaluated_candidates()) == 3


def test_search_history_completeness(
    mock_mutation_function,
    mock_scoring_function,
    starter_kernel_code,
    reference_kernel_code,
):
    """Test that search history contains all evaluated kernels."""
    mutation1 = MutatedKernel(
        kernel_code="def kernel(x): return x * 2  # fast",
        ancestor_ulid=None,
    )
    mutation2 = MutatedKernel(
        kernel_code="def kernel(x): return x * 2  # slow",
        ancestor_ulid=None,
    )
    mutation3 = MutatedKernel(
        kernel_code="def kernel(x): return x * 2  # fast",
        ancestor_ulid=mutation1.ulid,
    )
    mutation4 = MutatedKernel(
        kernel_code="def kernel(x): return x * 2  # slow",
        ancestor_ulid=mutation1.ulid,
    )

    mock_mutation_function.side_effect = [mutation1, mutation2, mutation3, mutation4]

    config = GreedySearchConfig(
        max_depth=2,
        num_mutations=2,
        starter_kernel_code=starter_kernel_code,
        reference_kernel_code=reference_kernel_code,
        mutation_function=mock_mutation_function,
        scoring_function=mock_scoring_function,
    )
    search = GreedySearch(config=config)

    result = search.run()

    # Verify starter is evaluated
    assert any(c.code == starter_kernel_code for c in result.evaluated_candidates())


def test_search_best_kernel_selection_logic(
    mock_mutation_function,
    mock_scoring_function,
    starter_kernel_code,
    reference_kernel_code,
):
    """Test that best kernel selection logic correctly picks highest speedup."""
    # Setup: mutations with different speedups
    mutation1 = MutatedKernel(
        kernel_code="def kernel(x): return x * 2  # slow",  # speedup 0.5
        ancestor_ulid=None,
    )
    mutation2 = MutatedKernel(
        kernel_code="def kernel(x): return x * 2",  # speedup 1.0
        ancestor_ulid=None,
    )
    mutation3 = MutatedKernel(
        kernel_code="def kernel(x): return x * 2  # fast",  # speedup 3.0
        ancestor_ulid=None,
    )

    mock_mutation_function.side_effect = [mutation1, mutation2, mutation3]

    config = GreedySearchConfig(
        max_depth=1,
        num_mutations=3,
        starter_kernel_code=starter_kernel_code,
        reference_kernel_code=reference_kernel_code,
        mutation_function=mock_mutation_function,
        scoring_function=mock_scoring_function,
    )
    search = GreedySearch(config=config)

    result = search.run()

    # Best should be mutation3 with highest speedup (3.0)
    assert len(result.rounds) == 1
    assert result.rounds[0].outcome.kind == "winner_selected"
    assert isinstance(result.best_evaluation, ValidEvaluation)
    assert result.best_evaluation.speedup == 3.0
    assert "fast" in result.best_candidate().code


def test_search_mutation_function_exception_handling(
    mock_mutation_function,
    mock_scoring_function,
    starter_kernel_code,
    reference_kernel_code,
):
    """Test that exceptions in mutation function are handled gracefully."""
    # Setup: mutation function raises exception for some calls
    mutation1 = MutatedKernel(
        kernel_code="def kernel(x): return x * 2  # fast",
        ancestor_ulid=None,
    )

    mock_mutation_function.side_effect = [
        Exception("LLM API error"),
        mutation1,  # Second call succeeds
    ]

    config = GreedySearchConfig(
        max_depth=1,
        num_mutations=2,
        starter_kernel_code=starter_kernel_code,
        reference_kernel_code=reference_kernel_code,
        mutation_function=mock_mutation_function,
        scoring_function=mock_scoring_function,
    )
    search = GreedySearch(config=config)

    result = search.run()

    # Should still complete successfully with the one valid mutation
    assert len(result.rounds) == 1
    assert isinstance(result.rounds[0].mutation_attempts[0], MutationFailure)
    assert len(result.evaluated_candidates()) == 2  # starter + 1 scored mutation


def test_search_scoring_function_exception_handling(
    mock_mutation_function,
    mock_scoring_function,
    starter_kernel_code,
    reference_kernel_code,
):
    """Test that exceptions in scoring function are handled gracefully."""
    mutation1 = MutatedKernel(
        kernel_code="def kernel(x): return x * 2  # fast",
        ancestor_ulid=None,
    )
    mutation2 = MutatedKernel(
        kernel_code="def kernel(x): return x * 2  # slow",
        ancestor_ulid=None,
    )

    mock_mutation_function.side_effect = [mutation1, mutation2]

    # Scoring function that raises exception for first call
    call_count = 0

    def scoring_with_exception(
        mutated_code: str, reference_code: str
    ) -> KernelScoringResult:
        nonlocal call_count
        call_count += 1
        # First call is scoring the starter kernel baseline inside GreedySearch.search().
        # Raise on the first mutation scoring call instead.
        if call_count == 2:
            raise RuntimeError("Scoring error")
        return mock_scoring_function(mutated_code, reference_code)

    config = GreedySearchConfig(
        max_depth=1,
        num_mutations=2,
        starter_kernel_code=starter_kernel_code,
        reference_kernel_code=reference_kernel_code,
        mutation_function=mock_mutation_function,
        scoring_function=scoring_with_exception,
    )
    search = GreedySearch(config=config)

    result = search.run()

    # Should still complete successfully with the one valid scored mutation
    # Evaluated candidates should contain starter + 1 successfully scored mutation
    assert len(result.rounds) == 1
    assert isinstance(result.rounds[0].scoring_attempts[0], ScoringFailure)
    assert len(result.evaluated_candidates()) == 2


def test_search_checkpoint_resume_between_rounds(
    starter_kernel_code, reference_kernel_code, mock_scoring_function
):
    """Checkpoint after 1 round, then resume to full depth."""
    # Make 6 mutations total (3 rounds * 2 mutations).
    mutations = [
        MutatedKernel(
            kernel_code="def kernel(x): return x * 2  # fast", ancestor_ulid=None
        ),
        MutatedKernel(
            kernel_code="def kernel(x): return x * 2  # slow", ancestor_ulid=None
        ),
        MutatedKernel(
            kernel_code="def kernel(x): return x * 3  # fast", ancestor_ulid=None
        ),
        MutatedKernel(
            kernel_code="def kernel(x): return x * 4  # slow", ancestor_ulid=None
        ),
        MutatedKernel(
            kernel_code="def kernel(x): return x * 5  # fast", ancestor_ulid=None
        ),
        MutatedKernel(
            kernel_code="def kernel(x): return x * 6  # slow", ancestor_ulid=None
        ),
    ]

    mock_mutation_full = MagicMock(spec=MutationFunction)
    mock_mutation_full.side_effect = list(mutations)

    mock_mutation_part = MagicMock(spec=MutationFunction)
    mock_mutation_part.side_effect = list(mutations)

    # Full run to depth 3.
    config_full = GreedySearchConfig(
        max_depth=3,
        num_mutations=2,
        starter_kernel_code=starter_kernel_code,
        reference_kernel_code=reference_kernel_code,
        mutation_function=mock_mutation_full,
        scoring_function=mock_scoring_function,
    )
    full = GreedySearch(config=config_full).run()

    # Partial run to depth 1.
    config_part = GreedySearchConfig(
        max_depth=1,
        num_mutations=2,
        starter_kernel_code=starter_kernel_code,
        reference_kernel_code=reference_kernel_code,
        mutation_function=mock_mutation_part,
        scoring_function=mock_scoring_function,
    )
    partial = GreedySearch(config=config_part).run()

    # Resume to depth 3 with the same mutation mock (continuing side effects).
    config_resume = GreedySearchConfig(
        max_depth=3,
        num_mutations=2,
        starter_kernel_code=starter_kernel_code,
        reference_kernel_code=reference_kernel_code,
        mutation_function=mock_mutation_part,
        scoring_function=mock_scoring_function,
    )
    resumed = GreedySearch(config=config_resume).resume(partial)

    assert len(full.rounds) == 3
    assert len(resumed.rounds) == 3

    assert isinstance(full.best_evaluation, ValidEvaluation)
    assert isinstance(resumed.best_evaluation, ValidEvaluation)
    assert full.best_evaluation.speedup == resumed.best_evaluation.speedup
    assert full.best_candidate().code == resumed.best_candidate().code

    # Compare selected parent codes per round (ULIDs differ across runs due to fresh starter ULID).
    for i in range(3):
        full_selected_code = full.candidates.get(
            full.rounds[i].selected_parent_ulid
        ).code
        resumed_selected_code = resumed.candidates.get(
            resumed.rounds[i].selected_parent_ulid
        ).code
        assert full.rounds[i].outcome.kind == resumed.rounds[i].outcome.kind
        assert full_selected_code == resumed_selected_code


def test_greedy_search_passes_parent_evaluation_into_mutation_context() -> None:
    contexts: list[MutationContext] = []
    mutation_codes = [
        "ok_candidate",
        "compile_fail_candidate",
        "ok_candidate_round2",
        "compile_fail_round2",
    ]

    def mutation_function(context: MutationContext) -> MutatedKernel:
        contexts.append(context)
        code = mutation_codes[len(contexts) - 1]
        return MutatedKernel(
            kernel_code=code,
            ancestor_ulid=context.previous_kernel_ulid,
        )

    def scoring_function(
        mutated_code: str, _reference_code: str
    ) -> KernelScoringResult:
        exec_result = KernelExecResult(
            compiled=True,
            correctness=True,
            runtime=10.0,
            ref_runtime=20.0,
            metadata={},
        )
        if "compile_fail" in mutated_code:
            exec_result.compiled = False
            exec_result.correctness = False
            exec_result.metadata = {
                "compilation_error_name": "CompilerError",
                "compilation_error": "Failed to compile",
            }
            return KernelScoringResult(
                exec_result=exec_result, speedup=0.0, is_valid=False
            )
        if "ok_candidate" in mutated_code:
            return KernelScoringResult(
                exec_result=exec_result, speedup=3.0, is_valid=True
            )
        return KernelScoringResult(exec_result=exec_result, speedup=2.0, is_valid=True)

    config = GreedySearchConfig(
        max_depth=2,
        num_mutations=2,
        starter_kernel_code="starter_ok",
        reference_kernel_code="reference_ok",
        mutation_function=mutation_function,
        scoring_function=scoring_function,
    )
    search = GreedySearch(config=config)
    result = search.run()

    assert len(contexts) == 4
    assert contexts[0].previous_evaluation is not None
    assert contexts[0].previous_evaluation.execution_feedback.kind == "success"
    assert contexts[2].previous_evaluation is not None
    assert contexts[2].previous_evaluation.execution_feedback.kind == "success"
    assert result.best_candidate().code.startswith("ok_candidate")


def test_greedy_search_feedback_driven_mutator_uses_compile_failed_feedback() -> None:
    def mutation_function(context: MutationContext) -> MutatedKernel:
        assert context.previous_evaluation is not None
        feedback_kind = context.previous_evaluation.execution_feedback.kind
        if feedback_kind == "compile_failed":
            code = "ok_candidate"
        else:
            code = "compile_fail_candidate"
        return MutatedKernel(
            kernel_code=code,
            ancestor_ulid=context.previous_kernel_ulid,
        )

    def scoring_function(
        mutated_code: str, _reference_code: str
    ) -> KernelScoringResult:
        exec_result = KernelExecResult(
            compiled=True,
            correctness=True,
            runtime=10.0,
            ref_runtime=20.0,
            metadata={},
        )
        if "compile_fail" in mutated_code:
            exec_result.compiled = False
            exec_result.correctness = False
            exec_result.metadata = {
                "compilation_error_name": "CompilerError",
                "compilation_error": "Failed to compile",
            }
            return KernelScoringResult(
                exec_result=exec_result, speedup=0.0, is_valid=False
            )
        return KernelScoringResult(exec_result=exec_result, speedup=2.0, is_valid=True)

    config = GreedySearchConfig(
        max_depth=1,
        num_mutations=1,
        starter_kernel_code="starter_compile_fail",
        reference_kernel_code="reference_ok",
        mutation_function=mutation_function,
        scoring_function=scoring_function,
    )
    search = GreedySearch(config=config)
    result = search.run()

    assert result.rounds[0].outcome.kind == "winner_selected"
    assert result.best_candidate().code == "ok_candidate"
