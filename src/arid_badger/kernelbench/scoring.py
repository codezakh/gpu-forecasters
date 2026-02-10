import torch
from arid_badger.kernelbench.core import KernelScoringResult
from arid_badger.kernelbench.eval_logged import eval_kernel_against_ref_logged
from kernelbench.eval import get_torch_dtype_from_string
from pathlib import Path
from typing import Optional


def score_kernel(
    mutated_kernel_code: str,
    reference_kernel_code: str,
    backend: str = "cuda",
    precision: str = "fp32",
    num_correct_trials: int = 5,
    num_perf_trials: int = 100,
    build_dir: Optional[Path] = None,
) -> KernelScoringResult:
    """
    Score a mutated kernel against a reference kernel.

    Args:
        mutated_kernel_code: Source code of the mutated kernel
        reference_kernel_code: Source code of the reference kernel
        backend: Backend type ("cuda", "triton", etc.)
        precision: Precision string ("fp32", "fp16", "bf16")
        num_correct_trials: Number of correctness trials
        num_perf_trials: Number of performance trials

    Returns:
        KernelScoringResult with exec_result, speedup, and validity flag
    """
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. KernelBench evaluation requires CUDA-capable GPU."
        )

    # Convert precision string to torch dtype
    torch_precision: torch.dtype = get_torch_dtype_from_string(precision)

    if build_dir is None:
        build_dir = Path("/tmp/arid_badger_torch_extensions")

    assert build_dir is not None

    # Evaluate kernel using KernelBench's evaluation function
    exec_result = eval_kernel_against_ref_logged(
        original_model_src=reference_kernel_code,
        custom_model_src=mutated_kernel_code,
        measure_performance=True,
        num_correct_trials=num_correct_trials,
        num_perf_trials=num_perf_trials,
        backend=backend,
        precision=torch_precision,
        timing_method="cuda_event",
        verbose=False,
        build_dir=build_dir,
    )
    if exec_result is None:
        # KernelBench returns None on transient lock/cache errors during concurrent compilation.
        # Callers that run parallel compilation should retry.
        raise RuntimeError(
            "KernelBench evaluation returned None (likely compile cache lock error); retry scoring."
        )

    # Calculate speedup (same logic as demo_kernelbench_apis.py)
    if not exec_result.correctness:
        return KernelScoringResult(exec_result=exec_result, speedup=0.0, is_valid=False)

    if exec_result.runtime <= 0 or exec_result.ref_runtime <= 0:
        return KernelScoringResult(exec_result=exec_result, speedup=0.0, is_valid=False)

    speedup: float = exec_result.ref_runtime / exec_result.runtime

    return KernelScoringResult(exec_result=exec_result, speedup=speedup, is_valid=True)
