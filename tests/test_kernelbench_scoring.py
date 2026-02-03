import math

import pytest
import torch
from kernelbench.prompt_constructor_toml import get_prompt_for_backend

from arid_badger.kernelbench.scoring import score_kernel


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
    result = score_kernel(
        mutated_kernel_code=MUTATED_KERNEL_CODE,
        reference_kernel_code=REFERENCE_KERNEL_CODE,
        backend="cuda",
        precision="fp32",
        num_correct_trials=1,
        num_perf_trials=5,
    )

    assert result.exec_result is not None
    assert (
        result.exec_result.compiled
    ), f"Kernel failed to compile. Metadata: {result.exec_result.metadata!r}"
    assert math.isfinite(result.speedup)
    assert not result.is_valid or result.exec_result.correctness
