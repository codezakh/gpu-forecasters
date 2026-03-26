"""Integration test for subprocess-isolated kernel scoring."""

import math
import re

import pytest
import torch
from kernelbench.dataset import BaseDataset, Problem, construct_kernelbench_dataset

from arid_badger.kernelbench.isolated_scoring import run_scoring_in_subprocess
from arid_badger.kernelbench.scoring import check_kernel_exec_result_valid
from arid_badger.typing_utils import is_ok


@pytest.mark.integration
@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_isolated_scoring_returns_valid_result() -> None:
    """run_scoring_in_subprocess should return a valid result for a known-good kernel."""
    dataset: BaseDataset = construct_kernelbench_dataset(
        level=1,
        source="local",
    )
    problem: Problem = dataset.get_problem_by_id(1)
    reference_kernel_code: str = problem.code

    mutated_kernel_code = re.sub(r"\bModel\b", "ModelNew", reference_kernel_code)

    try:
        outcome = run_scoring_in_subprocess(
            mutated_kernel_code=mutated_kernel_code,
            reference_kernel_code=reference_kernel_code,
            backend="cuda",
            precision="fp32",
            num_correct_trials=1,
            num_perf_trials=5,
        )
    except RuntimeError as exc:
        message = str(exc)
        if "returned None" in message or "compile cache lock" in message:
            pytest.skip(
                "KernelBench evaluation returned None (compile cache lock); rerun test."
            )
        raise

    assert is_ok(outcome), f"Scoring failed: {outcome.unwrap_err()}"
    exec_result = outcome.unwrap()
    assert exec_result.compiled is True
    assert exec_result.correctness is True
    assert exec_result.runtime > 0
    assert exec_result.ref_runtime > 0
    assert math.isfinite(exec_result.runtime)
    assert math.isfinite(exec_result.ref_runtime)

    is_valid = check_kernel_exec_result_valid(exec_result)
    assert is_valid is True
    speedup = exec_result.ref_runtime / exec_result.runtime
    assert speedup > 0
