"""Integration test for subprocess-isolated kernel scoring."""

import math

import pytest
import torch
from kernelbench.dataset import BaseDataset, Problem, construct_kernelbench_dataset

from arid_badger.kernelbench.scoring import score_kernel


@pytest.mark.integration
@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_isolated_scoring_returns_valid_result() -> None:
    """score_kernel(isolate=True) should return a valid result for a known-good kernel."""
    dataset: BaseDataset = construct_kernelbench_dataset(
        level=1,
        source="local",
    )
    problem: Problem = dataset.get_problem_by_id(1)
    reference_kernel_code: str = problem.code

    # The reference kernel uses "Model"; we need a "ModelNew" variant for the
    # mutated kernel.  For a simple identity test, just reuse the same logic
    # under the ModelNew name.  Use a regex word-boundary match to avoid
    # mangling "Module" → "ModelNewule".
    import re

    mutated_kernel_code = re.sub(r"\bModel\b", "ModelNew", reference_kernel_code)

    try:
        result = score_kernel(
            mutated_kernel_code=mutated_kernel_code,
            reference_kernel_code=reference_kernel_code,
            backend="cuda",
            precision="fp32",
            num_correct_trials=1,
            num_perf_trials=5,
            isolate=True,
        )
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
    assert result.speedup > 0
    assert result.is_valid is True
