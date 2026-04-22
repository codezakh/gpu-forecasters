"""Lifecycle and integration tests for the v2 Modal TriMul evaluation
provider.

Unit tests: __enter__ / __exit__ cleanup, submit-without-enter raises.

Integration tests (marked, opt-in): hit live Modal with one trivial
candidate and confirm we get a well-formed Evaluation back.
"""

from __future__ import annotations

import pytest

from arid_badger.max_reward_puct.v2.scoring_providers.trimul_modal import (
    TriMulModalProvider,
)
from arid_badger.trimul.cases import CORRECTNESS_CASES


def _trivial_kernel() -> str:
    """A correct-but-slow kernel: just calls the reference impl in
    PyTorch. Useful for end-to-end plumbing tests because it's
    extremely unlikely to crash the GPU process."""
    return '''
import torch
from torch import nn, einsum

class Reference(nn.Module):
    def __init__(self, dim, hidden_dim, weights):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.norm.weight = nn.Parameter(weights["norm.weight"])
        self.norm.bias = nn.Parameter(weights["norm.bias"])
        self.left_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.left_proj.weight = nn.Parameter(weights["left_proj.weight"])
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.right_proj.weight = nn.Parameter(weights["right_proj.weight"])
        self.left_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.left_gate.weight = nn.Parameter(weights["left_gate.weight"])
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.right_gate.weight = nn.Parameter(weights["right_gate.weight"])
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.out_gate.weight = nn.Parameter(weights["out_gate.weight"])
        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out_norm.weight = nn.Parameter(weights["to_out_norm.weight"])
        self.to_out_norm.bias = nn.Parameter(weights["to_out_norm.bias"])
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)
        self.to_out.weight = nn.Parameter(weights["to_out.weight"])

    def forward(self, x, mask):
        x = self.norm(x)
        left = self.left_proj(x)
        right = self.right_proj(x)
        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask
        left = left * self.left_gate(x).sigmoid()
        right = right * self.right_gate(x).sigmoid()
        out = einsum("... i k d, ... j k d -> ... i j d", left, right)
        out = self.to_out_norm(out)
        out = out * self.out_gate(x).sigmoid()
        return self.to_out(out)


def custom_kernel(data):
    input_tensor, mask, weights, config = data
    dim, hidden_dim = config["dim"], config["hidden_dim"]
    model = Reference(dim, hidden_dim, weights).to(input_tensor.device)
    with torch.no_grad():
        return model(input_tensor, mask).float()
'''


def test_submit_without_enter_raises():
    p = TriMulModalProvider(test_cases=CORRECTNESS_CASES[:1])
    with pytest.raises(RuntimeError, match="context manager"):
        _ = p.submit("# placeholder")


@pytest.mark.integration
def test_real_modal_call_returns_evaluation():
    """Live Modal call. Spawn a single trivial-but-correct candidate
    on a single small case, wait for the Evaluation, assert shape."""
    p = TriMulModalProvider(
        test_cases=CORRECTNESS_CASES[:1],
        gpu="A100-80GB",
        max_repeats=10,
        max_time_ns=5e9,
        max_in_flight=1,
    )
    with p:
        fut = p.submit(_trivial_kernel())
        evaluation = fut.result(timeout=900)

    # The evaluation can be success (reward set) or non-success (reward None
    # but observation populated with a feedback variant). We only assert
    # the shape is well-formed — the integration-level success of the
    # PyTorch reference path is incidental.
    assert evaluation is not None
    assert evaluation.observation is not None
    assert evaluation.observation.feedback is not None
