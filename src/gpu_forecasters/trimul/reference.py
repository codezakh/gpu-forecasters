"""TriMul reference implementation + input generator + oracle.

Vendored from ``ttt-discover/examples/gpu_mode/lib/bioml/trimul/reference.py``.
The upstream module imports ``input_t``/``output_t`` from a ``task.py``
sibling and ``DisableCuDNNTF32`` + ``make_match_reference`` from a
``utils.py`` sibling. We move the type aliases into this module and pull
the helpers from our vendored ``comparison`` module.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Tuple

import torch
from torch import einsum, nn

from gpu_forecasters.trimul.comparison import DisableCuDNNTF32, make_match_reference

# Upstream defines these as TypeVars bound to concrete tuple/tensor shapes in
# its ``task.py``. For our vendored scoring pipeline they're just type aliases
# — the runtime contract is enforced by ``generate_input`` and ``ref_kernel``.
input_t = Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor], Dict[str, Any]]
output_t = torch.Tensor


class TriMul(nn.Module):
    """Triangle Multiplicative Update (outgoing variant).

    Based on https://github.com/lucidrains/triangle-multiplicative-module.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        device: "torch.device | str" = "cuda",
        dtype: str = "",
    ) -> None:
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim, bias=False, device=device)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False, device=device)

        self.left_gate = nn.Linear(dim, hidden_dim, bias=False, device=device)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False, device=device)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False, device=device)

        self.to_out_norm = nn.LayerNorm(hidden_dim, device=device)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False, device=device)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """x: [bs, seq_len, seq_len, dim], mask: [bs, seq_len, seq_len]."""
        x = self.norm(x)

        left = self.left_proj(x)
        right = self.right_proj(x)

        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum("... i k d, ... j k d -> ... i j d", left, right)

        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)


def ref_kernel(data: input_t) -> output_t:
    """Reference TriMul forward pass.

    Args:
        data: Tuple of (input, mask, weights, config).
    """
    with DisableCuDNNTF32():
        input_tensor, mask, weights, config = data
        trimul = TriMul(
            dim=config["dim"],
            hidden_dim=config["hidden_dim"],
            device=input_tensor.device,
        )

        trimul.norm.weight = nn.Parameter(weights["norm.weight"])
        trimul.norm.bias = nn.Parameter(weights["norm.bias"])
        trimul.left_proj.weight = nn.Parameter(weights["left_proj.weight"])
        trimul.right_proj.weight = nn.Parameter(weights["right_proj.weight"])
        trimul.left_gate.weight = nn.Parameter(weights["left_gate.weight"])
        trimul.right_gate.weight = nn.Parameter(weights["right_gate.weight"])
        trimul.out_gate.weight = nn.Parameter(weights["out_gate.weight"])
        trimul.to_out_norm.weight = nn.Parameter(weights["to_out_norm.weight"])
        trimul.to_out_norm.bias = nn.Parameter(weights["to_out_norm.bias"])
        trimul.to_out.weight = nn.Parameter(weights["to_out.weight"])

        return trimul(input_tensor, mask)


def generate_input(
    seqlen: int,
    bs: int,
    dim: int,
    hiddendim: int,
    seed: int,
    nomask: bool,
    distribution: str,
) -> input_t:
    """Generate a single TriMul test case deterministically from ``seed``."""
    batch_size = bs
    seq_len = seqlen
    hidden_dim = hiddendim
    no_mask = nomask

    config: Dict[str, Any] = {"hidden_dim": hidden_dim, "dim": dim}

    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)

    weights: Dict[str, torch.Tensor] = {}

    if distribution == "cauchy":
        u = torch.empty(
            (batch_size, seq_len, seq_len, dim),
            device="cuda",
            dtype=torch.float32,
        )
        u.uniform_(0.0, 1.0, generator=gen)
        input_tensor = 2.0 * torch.tan(math.pi * (u - 0.5))
    else:
        input_tensor = torch.randn(
            (batch_size, seq_len, seq_len, dim),
            device="cuda",
            dtype=torch.float32,
            generator=gen,
        ).contiguous()

    if no_mask:
        mask = torch.ones(
            batch_size, seq_len, seq_len, device=input_tensor.device
        )
    else:
        mask = torch.randint(
            0,
            2,
            (batch_size, seq_len, seq_len),
            device=input_tensor.device,
            generator=gen,
        )

    weights["norm.weight"] = torch.randn(dim, device="cuda", dtype=torch.float32)
    weights["norm.bias"] = torch.randn(dim, device="cuda", dtype=torch.float32)
    weights["left_proj.weight"] = torch.randn(
        hidden_dim, dim, device="cuda", dtype=torch.float32
    ) / math.sqrt(hidden_dim)
    weights["right_proj.weight"] = torch.randn(
        hidden_dim, dim, device="cuda", dtype=torch.float32
    ) / math.sqrt(hidden_dim)
    weights["left_gate.weight"] = torch.randn(
        hidden_dim, dim, device="cuda", dtype=torch.float32
    ) / math.sqrt(hidden_dim)
    weights["right_gate.weight"] = torch.randn(
        hidden_dim, dim, device="cuda", dtype=torch.float32
    ) / math.sqrt(hidden_dim)
    weights["out_gate.weight"] = torch.randn(
        hidden_dim, dim, device="cuda", dtype=torch.float32
    ) / math.sqrt(hidden_dim)
    weights["to_out_norm.weight"] = torch.randn(
        hidden_dim, device="cuda", dtype=torch.float32
    )
    weights["to_out.weight"] = torch.randn(
        dim, hidden_dim, device="cuda", dtype=torch.float32
    ) / math.sqrt(dim)
    weights["to_out_norm.bias"] = torch.randn(
        hidden_dim, device="cuda", dtype=torch.float32
    )

    return (input_tensor, mask, weights, config)


# ``check_implementation`` signature: ``(data, submission_output) -> (good, msg)``.
# Uses rtol/atol = 2e-2 to match upstream's tolerance (bf16 intermediates in
# some reference candidates require a loose tolerance).
check_implementation: Callable[
    [input_t, torch.Tensor], Tuple[bool, str]
] = make_match_reference(ref_kernel, rtol=2e-2, atol=2e-2)
