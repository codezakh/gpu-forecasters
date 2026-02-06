import math

import pytest
import torch
from kernelbench.dataset import BaseDataset, Problem, construct_kernelbench_dataset
from kernelbench.eval import eval_kernel_against_ref as eval_kernel_against_ref_upstream

from arid_badger.kernelbench.eval_logged import eval_kernel_against_ref_logged


@pytest.mark.integration
@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_eval_logged_matches_upstream_timing():
    level: int = 1
    problem_id: int = 1
    backend: str = "cuda"
    precision: torch.dtype = torch.float32

    dataset: BaseDataset = construct_kernelbench_dataset(
        level=level,
        source="local",
    )
    problem: Problem = dataset.get_problem_by_id(problem_id)
    starter_kernel_code: str = problem.code
    custom_kernel_code = (
        f"{starter_kernel_code}\n\n"
        "# KernelBench expects ModelNew in custom_model_src; keep behavior identical\n"
        "# by inheriting from the reference Model without overriding anything.\n"
        "class ModelNew(Model):\n"
        "    pass\n"
    )

    upstream_result = eval_kernel_against_ref_upstream(
        original_model_src=starter_kernel_code,
        custom_model_src=custom_kernel_code,
        measure_performance=True,
        num_correct_trials=3,
        num_perf_trials=100,
        backend=backend,
        precision=precision,
        timing_method="cuda_event",
        verbose=False,
        build_dir=None,
    )
    if upstream_result is None:
        pytest.skip(
            "KernelBench returned None (likely compile cache lock). Rerun to compare."
        )

    logged_result = eval_kernel_against_ref_logged(
        original_model_src=starter_kernel_code,
        custom_model_src=custom_kernel_code,
        measure_performance=True,
        num_correct_trials=3,
        num_perf_trials=100,
        backend=backend,
        precision=precision,
        timing_method="cuda_event",
        verbose=False,
        build_dir=None,
    )
    if logged_result is None:
        pytest.skip(
            "Logged eval returned None (likely compile cache lock). Rerun to compare."
        )

    assert upstream_result.compiled == logged_result.compiled
    assert upstream_result.correctness == logged_result.correctness
    assert upstream_result.compiled is True
    assert upstream_result.correctness is True

    assert upstream_result.runtime > 0
    assert upstream_result.ref_runtime > 0
    assert logged_result.runtime > 0
    assert logged_result.ref_runtime > 0

    def rel_diff(a: float, b: float) -> float:
        return abs(a - b) / max(abs(a), abs(b))

    assert rel_diff(upstream_result.runtime, logged_result.runtime) <= 0.02
    assert rel_diff(upstream_result.ref_runtime, logged_result.ref_runtime) <= 0.02
    assert math.isfinite(logged_result.runtime)
    assert math.isfinite(logged_result.ref_runtime)
