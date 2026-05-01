"""End-to-end integration test for the v2 KernelBench Modal provider.

These tests open a real ``app.run()`` session against the
``arid-badger-kernel-split`` Modal app, run the chained
compile-on-CPU / benchmark-on-GPU pipeline asynchronously, and verify
that the resulting ``Evaluation`` shapes match what the unit tests
exercise via mocks.

Selected explicitly:

    uv run --env-file .env pytest -m modal \\
        tests/kernelbench/v2/providers/test_modal_scoring_integration.py -v
"""

from __future__ import annotations

import time

import pytest

from arid_badger.kernelbench.core import (
    CompileFailedFeedback,
    InfrastructureFailureFeedback,
    SuccessFeedback,
)
from arid_badger.kernelbench.v2.providers.modal_scoring import (
    KernelBenchModalProvider,
)
from arid_badger.modal_gpu import GpuKind

# Same 128x128 matmul fixture used by the v1 modal scoring integration
# tests. The point here is to exercise plumbing — provider lifecycle,
# semaphore-bounded async dispatch, failure wrapping — not to demonstrate
# kernel speedups, so a tiny problem size that finishes quickly is ideal.

_REFERENCE_KERNEL_CODE = """
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

_CORRECT_KERNEL_CODE = """
import torch
import torch.nn as nn


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.mm(a, b)
"""

# Wrong class name → KernelBench's loader cannot find ``ModelNew``,
# surfaces as a kernel-defect failure (compile or runtime depending on
# stage). Same fixture used by the v1 ``test_modal_scoring_integration``.
_BROKEN_KERNEL_CODE = """
import torch
import torch.nn as nn


class WrongName(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.mm(a, b)
"""


@pytest.mark.integration
@pytest.mark.modal
@pytest.mark.slow
def test_v2_provider_submit_correct_kernel_end_to_end() -> None:
    """A correct kernel returns an Evaluation with SuccessFeedback and
    a positive speedup reward. Exercises the full Modal chain:
    ``app.run()`` → CPU compile → GPU bench → wrap → resolve future."""
    with KernelBenchModalProvider(
        reference_kernel_code=_REFERENCE_KERNEL_CODE,
        gpu=GpuKind.L4,
        num_correct_trials=1,
        num_perf_trials=5,
        max_in_flight=2,
    ) as provider:
        future = provider.submit(_CORRECT_KERNEL_CODE)
        evaluation = future.result(timeout=300.0)

    assert evaluation.reward is not None, (
        f"Expected non-None reward for a correct kernel; got {evaluation}"
    )
    assert evaluation.reward > 0, f"Expected positive speedup, got {evaluation.reward}"
    assert isinstance(evaluation.observation.feedback, SuccessFeedback)


@pytest.mark.integration
@pytest.mark.modal
@pytest.mark.slow
def test_v2_provider_submit_broken_kernel_surfaces_failure() -> None:
    """A broken kernel (missing ``ModelNew``) must surface as either
    ``CompileFailedFeedback`` or ``InfrastructureFailureFeedback`` —
    never as a ``SuccessFeedback`` with a positive reward."""
    with KernelBenchModalProvider(
        reference_kernel_code=_REFERENCE_KERNEL_CODE,
        gpu=GpuKind.L4,
        num_correct_trials=1,
        num_perf_trials=5,
        max_in_flight=2,
    ) as provider:
        future = provider.submit(_BROKEN_KERNEL_CODE)
        evaluation = future.result(timeout=300.0)

    assert evaluation.reward is None, (
        f"Expected None reward for a broken kernel; got {evaluation}"
    )
    assert isinstance(
        evaluation.observation.feedback,
        (CompileFailedFeedback, InfrastructureFailureFeedback),
    ), f"Got unexpected feedback type: {type(evaluation.observation.feedback)}"


@pytest.mark.integration
@pytest.mark.modal
@pytest.mark.slow
def test_v2_provider_concurrent_submits_share_session() -> None:
    """Multiple concurrent submits run in parallel against the same
    session.

    Two submits with ``max_in_flight=2`` should both start their CPU
    compiles immediately and the two GPU benchmarks should overlap. We
    verify behavior (both succeed, both return SuccessFeedback) rather
    than wall-clock — wall-clock is too flaky for assertions because
    Modal cold-starts dominate. The qualitative claim "concurrent
    dispatch works" is best confirmed by ``modal container list``
    during the run; this test guards the easier invariant: two
    concurrent submits resolve correctly without serialization
    deadlocks or shared-state corruption.
    """
    with KernelBenchModalProvider(
        reference_kernel_code=_REFERENCE_KERNEL_CODE,
        gpu=GpuKind.L4,
        num_correct_trials=1,
        num_perf_trials=5,
        max_in_flight=2,
    ) as provider:
        t0 = time.perf_counter()
        futures = [
            provider.submit(_CORRECT_KERNEL_CODE),
            provider.submit(_CORRECT_KERNEL_CODE),
        ]
        evaluations = [f.result(timeout=300.0) for f in futures]
        elapsed = time.perf_counter() - t0

    print(
        f"\n[concurrent] 2 submits resolved in {elapsed:.1f}s "
        f"(serial would be roughly 2× a single submit)"
    )

    assert len(evaluations) == 2
    for ev in evaluations:
        assert ev.reward is not None and ev.reward > 0
        assert isinstance(ev.observation.feedback, SuccessFeedback)
