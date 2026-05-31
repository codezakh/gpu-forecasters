from pathlib import Path
from typing import Optional

import torch
from arid_badger.kernelbench.core import KernelScoringResult
from arid_badger.kernelbench.eval_logged import eval_kernel_against_ref_logged
from kernelbench.eval import get_torch_dtype_from_string, KernelExecResult

DEFAULT_BUILD_DIR = Path("/tmp/arid_badger_torch_extensions")


def check_kernel_exec_result_valid(exec_result: KernelExecResult) -> bool:
    if not exec_result.correctness:
        return False
    if exec_result.runtime <= 0 or exec_result.ref_runtime <= 0:
        return False
    return True


def _score_kernel_impl(
    mutated_kernel_code: str,
    reference_kernel_code: str,
    backend: str = "cuda",
    precision: str = "fp32",
    num_correct_trials: int = 5,
    num_perf_trials: int = 100,
    build_dir: Optional[Path] = None,
) -> Optional[KernelExecResult]:
    """In-process scoring logic. Called directly or from a subprocess worker."""
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. KernelBench evaluation requires CUDA-capable GPU."
        )

    # Convert precision string to torch dtype
    torch_precision: torch.dtype = get_torch_dtype_from_string(precision)

    if build_dir is None:
        build_dir = DEFAULT_BUILD_DIR

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

    return exec_result
