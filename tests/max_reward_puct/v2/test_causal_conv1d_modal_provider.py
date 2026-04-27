"""Lifecycle and integration tests for the v2 Modal causal conv1d
evaluation provider.

Unit tests: ``submit`` without ``__enter__`` raises.

Integration tests (marked, opt-in): hit live Modal with one trivial
candidate and confirm we get a well-formed Evaluation back. Mirror of
``test_trimul_modal_provider``.
"""

from __future__ import annotations

import pytest

from arid_badger.causal_conv1d.cases import CORRECTNESS_CASES
from arid_badger.max_reward_puct.v2.scoring_providers.causal_conv1d_modal import (
    CausalConv1dModalProvider,
)


def _trivial_kernel() -> str:
    """A correct-but-slow kernel: just the PyTorch reference. Useful
    for end-to-end plumbing tests because it's extremely unlikely to
    crash the GPU process."""
    return '''
import torch
import torch.nn.functional as F


def custom_kernel(data):
    x, weight, bias = data
    _, D, _ = x.shape
    W = weight.shape[1]
    x_padded = F.pad(x, (W - 1, 0))
    return F.conv1d(
        x_padded,
        weight.unsqueeze(1),
        bias=bias,
        groups=D,
    )
'''


def test_submit_without_enter_raises() -> None:
    p = CausalConv1dModalProvider(test_cases=CORRECTNESS_CASES[:1])
    with pytest.raises(RuntimeError, match="context manager"):
        _ = p.submit("# placeholder")


@pytest.mark.integration
def test_real_modal_call_returns_evaluation() -> None:
    """Live Modal call. Spawn a single trivial-but-correct candidate
    on a single small case, wait for the Evaluation, assert shape."""
    p = CausalConv1dModalProvider(
        test_cases=CORRECTNESS_CASES[:1],
        gpu="A100-80GB",
        max_repeats=10,
        max_time_ns=5e9,
        max_in_flight=1,
    )
    with p:
        fut = p.submit(_trivial_kernel())
        evaluation = fut.result(timeout=900)

    # Evaluation can be success (reward set) or non-success (reward None
    # but observation populated with a feedback variant). Only assert
    # the shape is well-formed — the integration-level success of the
    # PyTorch reference path is incidental.
    assert evaluation is not None
    assert evaluation.observation is not None
    assert evaluation.observation.feedback is not None
