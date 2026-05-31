import math

import pytest
import torch

from gpu_forecasters.kernelbench.isolated_scoring import run_scoring_in_subprocess
from gpu_forecasters.kernelbench.scoring import check_kernel_exec_result_valid
from gpu_forecasters.typing_utils import is_ok


REFERENCE_KERNEL_CODE = """
import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)


N = 128


def get_inputs():
    a = torch.rand(N, N)
    b = torch.rand(N, N)
    return [a, b]


def get_init_inputs():
    return []
"""

MUTATED_KERNEL_CODE = """
import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.mm(a, b)
"""


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_kernelbench_scoring_hardcoded_kernels() -> None:
    outcome = run_scoring_in_subprocess(
        mutated_kernel_code=MUTATED_KERNEL_CODE,
        reference_kernel_code=REFERENCE_KERNEL_CODE,
        backend="cuda",
        precision="fp32",
        num_correct_trials=1,
        num_perf_trials=5,
    )

    assert is_ok(outcome), f"Scoring failed: {outcome.unwrap_err()}"
    exec_result = outcome.unwrap()
    assert exec_result.compiled, f"Kernel failed to compile. Metadata: {exec_result.metadata!r}"

    is_valid = check_kernel_exec_result_valid(exec_result)
    if is_valid:
        speedup = exec_result.ref_runtime / exec_result.runtime
        assert math.isfinite(speedup)
        assert exec_result.correctness
