"""
Integration test for KernelBench greedy search pipeline.

This test validates the entire pipeline:
1. Loads a prompt and problem from the KernelBench dataset
2. Uses MutationFunction to generate a mutated kernel via LLM
3. Scores the kernel and asserts compilation + measurable runtime ratio

This test requires:
- CUDA-capable GPU
- GEMINI_API_KEY environment variable set (for Gemini 3 Flash Preview)
- Run with: pytest -m integration
"""

import math

import pytest
import torch
from kernelbench.dataset import BaseDataset, Problem, construct_kernelbench_dataset
from kernelbench.prompt_constructor_toml import get_prompt_for_backend

from arid_badger.greedy_search.components import (
    MutationContext,
    MutationFunction,
    MutatedKernel,
)
from arid_badger.kernelbench.core import KernelScoringResult
from arid_badger.kernelbench.scoring import score_kernel


@pytest.mark.integration
@pytest.mark.cuda
@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_kernelbench_greedy_search_pipeline():
    """
    Integration test for the full KernelBench greedy search pipeline.

    This test:
    1. Loads the starter kernel (level 1, problem 1) from KernelBench dataset
    2. Builds the default prompt using KernelBench's prompt constructor
    3. Generates a mutated kernel using MutationFunction with Gemini 3 Flash Preview
    4. Scores the mutated kernel against the reference
    5. Asserts that:
       - The kernel compiles successfully
       - Both mutated and reference kernels have positive runtimes
       - The speedup ratio is finite (allowing estimation of speedup/slowdown)
    """
    # Configuration matching demo_kernelbench_apis.py defaults
    level: int = 1
    problem_id: int = 1
    backend: str = "cuda"
    precision: str = "fp32"
    prompt_option: str = "one_shot"

    # 1. Load KernelBench data (level 1, problem 1)
    dataset: BaseDataset = construct_kernelbench_dataset(
        level=level,
        source="local",
    )
    problem: Problem = dataset.get_problem_by_id(problem_id)
    starter_kernel_code: str = problem.code

    # 2. Build default prompt (same as demo_kernelbench_apis.py)
    prompt = get_prompt_for_backend(
        ref_arch_src=starter_kernel_code,
        backend=backend,
        option=prompt_option,
        precision=precision,
        include_hardware=False,
        gpu_name=None,
    )

    # 3. Create mutation context
    # We're mutating from the starter kernel (no previous mutation)
    # So we set previous_kernel_code to the starter, but no previous_kernel_ulid
    context = MutationContext(
        previous_kernel_code=starter_kernel_code,
        previous_kernel_ulid=None,  # Starter kernel doesn't have a ulid
        prompt=prompt,
        ref_arch_src=starter_kernel_code,
        backend=backend,
        precision=precision,
    )

    # 4. Create MutationFunction with Gemini 3 Flash Preview (faster for tests)
    mutation_fn = MutationFunction(
        model="gemini/gemini-3-flash-preview",
        backend=backend,
        option=prompt_option,
        precision=precision,
    )

    # 5. Generate mutated kernel
    mutated_kernel: MutatedKernel = mutation_fn(context)

    # 6. Validate mutation structure
    assert mutated_kernel.ulid, "MutatedKernel should have a non-empty ulid"
    assert mutated_kernel.kernel_code, "MutatedKernel should have non-empty kernel_code"
    assert (
        mutated_kernel.ancestor_ulid == context.previous_kernel_ulid
    ), "ancestor_ulid should propagate from context.previous_kernel_ulid"

    # 7. Score the mutated kernel against the reference
    # Use fewer trials for faster test execution (integration test only needs to verify pipeline works)
    scoring_result: KernelScoringResult = score_kernel(
        mutated_kernel_code=mutated_kernel.kernel_code,
        reference_kernel_code=starter_kernel_code,
        backend=backend,
        precision=precision,
        num_correct_trials=1,  # Reduced for speed
        num_perf_trials=5,  # Reduced from 100 to 5 for faster test execution
    )

    # 8. Validate scoring result - core assertions
    assert (
        scoring_result.exec_result is not None
    ), "ScoringResult should have exec_result"

    # Assert kernel compiles successfully
    assert (
        scoring_result.exec_result.compiled
    ), f"Kernel failed to compile. Metadata: {scoring_result.exec_result.metadata!r}"

    # Assert both kernels have positive runtimes (allows speedup/slowdown estimation)
    assert (
        scoring_result.exec_result.runtime > 0
    ), f"Mutated kernel runtime should be positive, got {scoring_result.exec_result.runtime}"
    assert (
        scoring_result.exec_result.ref_runtime > 0
    ), f"Reference kernel runtime should be positive, got {scoring_result.exec_result.ref_runtime}"

    # Assert speedup ratio is finite (allows meaningful speedup/slowdown estimation)
    assert math.isfinite(scoring_result.speedup), (
        f"Speedup should be finite, got {scoring_result.speedup}. "
        f"Runtime: {scoring_result.exec_result.runtime}, "
        f"Ref runtime: {scoring_result.exec_result.ref_runtime}"
    )

    # Print diagnostic information
    print("[Integration Test] Scoring complete!")
    print(f"[Integration Test] Correctness: {scoring_result.exec_result.correctness}")
    print(f"[Integration Test] Compiled: {scoring_result.exec_result.compiled}")
    print(f"[Integration Test] Speedup: {scoring_result.speedup:.4f}x")
    print(f"[Integration Test] Is valid: {scoring_result.is_valid}")
    print(
        f"[Integration Test] Mutated kernel runtime: {scoring_result.exec_result.runtime:.2f} μs"
    )
    print(
        f"[Integration Test] Reference kernel runtime: {scoring_result.exec_result.ref_runtime:.2f} μs"
    )
