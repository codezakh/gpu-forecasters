import math

import pytest
import torch
from kernelbench.dataset import BaseDataset, Problem, construct_kernelbench_dataset

from arid_badger.greedy_search.domain import MutationContext
from arid_badger.greedy_search.feedback_mutation import (
    KernelBenchExecutionFeedbackMutationFunction,
)
from arid_badger.greedy_search.kernelbench_prompt_mutation import (
    KernelBenchPromptMutationFunction,
)
from arid_badger.greedy_search.search import GreedySearch, GreedySearchConfig
from arid_badger.kernelbench.scoring import score_kernel


@pytest.mark.integration
@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_feedback_mutation_success_path_runs_end_to_end() -> None:
    level = 1
    problem_id = 1
    dataset: BaseDataset = construct_kernelbench_dataset(
        level=level,
        source="local",
    )
    problem: Problem = dataset.get_problem_by_id(problem_id)
    starter_kernel_code: str = problem.code

    scoring_result = score_kernel(
        mutated_kernel_code=starter_kernel_code,
        reference_kernel_code=starter_kernel_code,
        backend="cuda",
        precision="fp32",
        num_correct_trials=1,
        num_perf_trials=5,
    )

    config = GreedySearchConfig(
        max_depth=1,
        num_mutations=1,
        starter_kernel_code=starter_kernel_code,
        reference_kernel_code=starter_kernel_code,
        mutation_function=KernelBenchPromptMutationFunction(),
        scoring_function=score_kernel,
    )
    evaluation = GreedySearch(config=config)._score_to_evaluation(scoring_result)

    context = MutationContext(
        reference_kernel_code=starter_kernel_code,
        previous_kernel_code=starter_kernel_code,
        previous_kernel_ulid=None,
        previous_evaluation=evaluation,
    )
    mutated = KernelBenchExecutionFeedbackMutationFunction()(context)

    scoring_result = score_kernel(
        mutated_kernel_code=mutated.kernel_code,
        reference_kernel_code=starter_kernel_code,
        backend="cuda",
        precision="fp32",
        num_correct_trials=1,
        num_perf_trials=5,
    )

    assert scoring_result.exec_result is not None
    assert scoring_result.exec_result.compiled
    assert math.isfinite(scoring_result.speedup)


@pytest.mark.integration
@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_feedback_mutation_compile_failed_path_runs_end_to_end() -> None:
    level = 1
    problem_id = 1
    dataset: BaseDataset = construct_kernelbench_dataset(
        level=level,
        source="local",
    )
    problem: Problem = dataset.get_problem_by_id(problem_id)
    starter_kernel_code: str = problem.code

    broken_kernel_code = "def not_model():\n    return 1\n"
    scoring_result = score_kernel(
        mutated_kernel_code=broken_kernel_code,
        reference_kernel_code=starter_kernel_code,
        backend="cuda",
        precision="fp32",
        num_correct_trials=1,
        num_perf_trials=5,
    )
    assert scoring_result.exec_result is not None
    assert scoring_result.exec_result.compiled is False

    config = GreedySearchConfig(
        max_depth=1,
        num_mutations=1,
        starter_kernel_code=starter_kernel_code,
        reference_kernel_code=starter_kernel_code,
        mutation_function=KernelBenchPromptMutationFunction(),
        scoring_function=score_kernel,
    )
    evaluation = GreedySearch(config=config)._score_to_evaluation(scoring_result)

    context = MutationContext(
        reference_kernel_code=starter_kernel_code,
        previous_kernel_code=broken_kernel_code,
        previous_kernel_ulid=None,
        previous_evaluation=evaluation,
    )
    mutated = KernelBenchExecutionFeedbackMutationFunction()(context)

    scoring_result = score_kernel(
        mutated_kernel_code=mutated.kernel_code,
        reference_kernel_code=starter_kernel_code,
        backend="cuda",
        precision="fp32",
        num_correct_trials=1,
        num_perf_trials=5,
    )

    assert scoring_result.exec_result is not None
