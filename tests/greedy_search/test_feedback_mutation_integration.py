import math

import pytest
import torch
from kernelbench.dataset import BaseDataset, Problem, construct_kernelbench_dataset

from gpu_forecasters.greedy_search.domain import KernelCandidate, MutationContext
from gpu_forecasters.greedy_search.feedback_mutation import (
    KernelBenchExecutionFeedbackMutationFunction,
)
from gpu_forecasters.greedy_search.scoring_provider import SerialScoringProvider
from gpu_forecasters.kernelbench.core import KernelScoringResult
from gpu_forecasters.kernelbench.isolated_scoring import run_scoring_in_subprocess
from gpu_forecasters.kernelbench.scoring import check_kernel_exec_result_valid


def _score_kernel(
    mutated_kernel_code: str,
    reference_kernel_code: str,
    backend: str = "cuda",
    precision: str = "fp32",
    num_correct_trials: int = 5,
    num_perf_trials: int = 100,
) -> KernelScoringResult:
    result = run_scoring_in_subprocess(
        mutated_kernel_code=mutated_kernel_code,
        reference_kernel_code=reference_kernel_code,
        backend=backend,
        precision=precision,
        num_correct_trials=num_correct_trials,
        num_perf_trials=num_perf_trials,
    )
    if result.is_err():
        raise RuntimeError(f"Scoring failed: {result.unwrap_err().reason}")
    exec_result = result.unwrap()
    is_valid = check_kernel_exec_result_valid(exec_result)
    speedup = exec_result.ref_runtime / exec_result.runtime if is_valid else float("nan")
    return KernelScoringResult(exec_result=exec_result, speedup=speedup, is_valid=is_valid)


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

    provider = SerialScoringProvider(
        scoring_function=lambda code, ref: _score_kernel(
            mutated_kernel_code=code,
            reference_kernel_code=ref,
            backend="cuda",
            precision="fp32",
            num_correct_trials=1,
            num_perf_trials=5,
        )
    )

    result = provider.score_reference(starter_kernel_code)
    assert result.is_ok(), f"Expected Ok, got Err: {result.unwrap_err()}"
    evaluation = result.unwrap()

    context = MutationContext(
        reference_kernel_code=starter_kernel_code,
        previous_kernel_code=starter_kernel_code,
        previous_kernel_ulid=None,
        previous_evaluation=evaluation,
    )
    mutated = KernelBenchExecutionFeedbackMutationFunction()(context)

    scoring_result = _score_kernel(
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

    provider = SerialScoringProvider(
        scoring_function=lambda code, ref: _score_kernel(
            mutated_kernel_code=code,
            reference_kernel_code=ref,
            backend="cuda",
            precision="fp32",
            num_correct_trials=1,
            num_perf_trials=5,
        )
    )

    broken_kernel_code = "def not_model():\n    return 1\n"
    broken_candidate = KernelCandidate(code=broken_kernel_code)
    _attempts, scored = provider.score_candidates([broken_candidate], starter_kernel_code)
    assert len(scored) == 1
    _, evaluation = scored[0]

    context = MutationContext(
        reference_kernel_code=starter_kernel_code,
        previous_kernel_code=broken_kernel_code,
        previous_kernel_ulid=None,
        previous_evaluation=evaluation,
    )
    mutated = KernelBenchExecutionFeedbackMutationFunction()(context)

    scoring_result = _score_kernel(
        mutated_kernel_code=mutated.kernel_code,
        reference_kernel_code=starter_kernel_code,
        backend="cuda",
        precision="fp32",
        num_correct_trials=1,
        num_perf_trials=5,
    )

    assert scoring_result.exec_result is not None
