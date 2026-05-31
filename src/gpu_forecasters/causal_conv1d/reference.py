"""Causal conv1d reference implementation + input generator + oracle.

Vendored from
``gpu-mode/reference-kernels/problems/helion/causal_conv1d_py/reference.py``.
The upstream module imports ``input_t``/``output_t`` from a ``task.py``
sibling and ``DeterministicContext`` + ``make_match_reference`` from a
``utils.py`` sibling. We move the type aliases into this module and pull
the helpers from our vendored ``comparison`` module.

Tolerance: ``rtol=1e-3, atol=1e-3`` per upstream.

Mirror of ``gpu_forecasters.trimul.reference``.
"""

from __future__ import annotations

from typing import Callable, Tuple

import torch
import torch.nn.functional as F

from gpu_forecasters.causal_conv1d.comparison import (
    DeterministicContext,
    make_match_reference,
)

# Upstream defines these as TypeVars bound to concrete tuple/tensor shapes in
# its ``task.py``. For our vendored scoring pipeline they're just type
# aliases — the runtime contract is enforced by ``generate_input`` and
# ``ref_kernel``.
input_t = Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
output_t = torch.Tensor


def generate_input(
    B: int, D: int, S: int, W: int, seed: int
) -> input_t:
    """Generate one causal-conv1d test case deterministically from ``seed``.

    Returns ``(x, weight, bias)`` where:
    - ``x``      : ``[B, D, S]`` float32
    - ``weight`` : ``[D, W]`` float32
    - ``bias``   : ``[D]`` float32
    """
    gen = torch.Generator(device="cuda")
    gen.manual_seed(seed)
    x = torch.randn(B, D, S, dtype=torch.float32, device="cuda", generator=gen).contiguous()
    weight = torch.randn(D, W, dtype=torch.float32, device="cuda", generator=gen).contiguous()
    bias = torch.randn(D, dtype=torch.float32, device="cuda", generator=gen).contiguous()
    return x, weight, bias


def ref_kernel(data: input_t) -> output_t:
    """Causal depthwise 1D conv reference.

    For each (b, d, t):
      out[b, d, t] = bias[d] + sum_k weight[d, k] * x[b, d, t - W + 1 + k]
    where out-of-bounds reads are zero (causal left-pad).
    """
    with DeterministicContext():
        x, weight, bias = data
        _, D, _ = x.shape
        W = weight.shape[1]

        # Causal (left) zero-padding so output[t] depends on input[t-W+1:t+1].
        x_padded = F.pad(x, (W - 1, 0))

        # Depthwise conv1d (groups=D).
        return F.conv1d(
            x_padded,
            weight.unsqueeze(1),  # [D, 1, W]
            bias=bias,
            groups=D,
        )


# ``check_implementation`` signature: ``(data, submission_output) -> (good, msg)``.
# Tolerance from upstream ``reference.py``: rtol=1e-3, atol=1e-3.
check_implementation: Callable[
    [input_t, torch.Tensor], Tuple[bool, str]
] = make_match_reference(ref_kernel, rtol=1e-3, atol=1e-3)
