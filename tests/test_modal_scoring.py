"""Integration tests for Modal-based kernel scoring.

These tests require Modal authentication (set up via `modal token new`) and
an active network connection to Modal's servers. They are excluded from the
default test run and must be selected explicitly:

    uv run --env-file .env pytest -m modal tests/test_modal_scoring.py -v
"""

import pytest

from arid_badger.kernelbench.modal_scoring import (
    modal_scoring_session,
    run_scoring_on_modal,
)
from arid_badger.hill_climbing.scoring_providers.kernelbench_modal import ModalProvider
from arid_badger.typing_utils import is_ok, is_err

# ---------------------------------------------------------------------------
# Test fixtures — same 128x128 matmul kernels used in test_kernelbench_scoring
# ---------------------------------------------------------------------------

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

# Correct optimized kernel (torch.mm is equivalent to torch.matmul for 2D)
CORRECT_KERNEL_CODE = """
import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.mm(a, b)
"""

# Broken kernel — wrong class name so KernelBench can't find ModelNew
BROKEN_KERNEL_CODE = """
import torch
import torch.nn as nn


class WrongName(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.mm(a, b)
"""


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.modal
@pytest.mark.slow
def test_modal_scoring_correct_kernel() -> None:
    """Correct kernel scores successfully on Modal."""
    with modal_scoring_session(gpu="L4", num_correct_trials=1, num_perf_trials=5) as score:
        result = score(CORRECT_KERNEL_CODE, REFERENCE_KERNEL_CODE)

    assert is_ok(result), f"Expected Ok, got Err: {result}"
    exec_result = result.unwrap()
    assert exec_result.compiled, f"Kernel failed to compile: {exec_result.metadata}"
    assert exec_result.correctness, f"Kernel produced incorrect output: {exec_result.metadata}"
    assert exec_result.runtime > 0, f"Expected positive runtime, got {exec_result.runtime}"
    assert exec_result.ref_runtime > 0, f"Expected positive ref_runtime, got {exec_result.ref_runtime}"


@pytest.mark.integration
@pytest.mark.modal
@pytest.mark.slow
def test_modal_scoring_broken_kernel() -> None:
    """Broken kernel (wrong class name) results in a compile/runtime failure."""
    with modal_scoring_session(gpu="L4", num_correct_trials=1, num_perf_trials=5) as score:
        result = score(BROKEN_KERNEL_CODE, REFERENCE_KERNEL_CODE)

    # Either the infrastructure reported an error (Err) or KernelBench
    # returned a KernelExecResult with compiled=False / correctness=False.
    if is_ok(result):
        exec_result = result.unwrap()
        assert not exec_result.correctness or not exec_result.compiled, (
            "Expected broken kernel to fail, but got a successful result"
        )
    else:
        assert is_err(result)


@pytest.mark.integration
@pytest.mark.modal
@pytest.mark.slow
def test_run_scoring_on_modal_convenience() -> None:
    """One-shot run_scoring_on_modal scores a correct kernel."""
    result = run_scoring_on_modal(
        CORRECT_KERNEL_CODE,
        REFERENCE_KERNEL_CODE,
        gpu="L4",
        num_correct_trials=1,
        num_perf_trials=5,
    )

    assert is_ok(result), f"Expected Ok, got Err: {result}"
    exec_result = result.unwrap()
    assert exec_result.compiled
    assert exec_result.correctness


@pytest.mark.integration
@pytest.mark.modal
@pytest.mark.slow
def test_modal_provider_evaluate() -> None:
    """ModalProvider.evaluate() returns a valid Evaluation via the EvaluationProvider protocol."""
    from arid_badger.kernelbench.core import SuccessFeedback

    with ModalProvider(
        reference_kernel_code=REFERENCE_KERNEL_CODE,
        gpu="L4",
        num_correct_trials=1,
        num_perf_trials=5,
    ) as provider:
        evaluation = provider.evaluate(CORRECT_KERNEL_CODE)

    assert evaluation.reward is not None, "Expected a non-None reward for a correct kernel"
    assert evaluation.reward > 0, f"Expected positive speedup, got {evaluation.reward}"
    assert isinstance(evaluation.observation.feedback, SuccessFeedback)
