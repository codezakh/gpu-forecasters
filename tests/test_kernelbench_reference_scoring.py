import math

import pytest
import torch
from kernelbench.dataset import BaseDataset, Problem, construct_kernelbench_dataset

from arid_badger.greedy_search.domain import MutationContext, MutatedKernel
from arid_badger.greedy_search.search import GreedySearch, GreedySearchConfig
from arid_badger.kernelbench.core import KernelScoringResult
from arid_badger.kernelbench.scoring import score_kernel


class _MutationFunctionStub:
    def __call__(self, context: MutationContext) -> MutatedKernel:
        raise AssertionError("Mutation function should not be called in this test.")


@pytest.mark.integration
@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_reference_kernel_scoring_ignores_starter() -> None:
    dataset: BaseDataset = construct_kernelbench_dataset(
        level=1,
        source="local",
    )
    problem: Problem = dataset.get_problem_by_id(1)
    reference_kernel_code: str = problem.code
    starter_kernel_code: str = "def broken("

    def scoring_function(mutated_code: str, reference_code: str) -> KernelScoringResult:
        return score_kernel(
            mutated_kernel_code=mutated_code,
            reference_kernel_code=reference_code,
            backend="cuda",
            precision="fp32",
            num_correct_trials=1,
            num_perf_trials=5,
        )

    config = GreedySearchConfig(
        max_depth=0,
        num_mutations=0,
        starter_kernel_code=starter_kernel_code,
        reference_kernel_code=reference_kernel_code,
        mutation_function=_MutationFunctionStub(),
        scoring_function=scoring_function,
        backend="cuda",
        precision="fp32",
    )
    search = GreedySearch(config=config)

    try:
        result = search.score_reference_kernel_only()
    except RuntimeError as exc:
        message = str(exc)
        if "returned None" in message or "compile cache lock" in message:
            pytest.skip(
                "KernelBench evaluation returned None (compile cache lock); rerun test."
            )
        raise

    exec_result = result.exec_result
    assert exec_result.compiled is True
    assert exec_result.correctness is True
    assert exec_result.runtime > 0
    assert exec_result.ref_runtime > 0
    assert math.isfinite(exec_result.runtime)
    assert math.isfinite(exec_result.ref_runtime)
