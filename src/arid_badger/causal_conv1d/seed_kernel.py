"""Seed program for causal conv1d evolutionary search.

A trivial PyTorch implementation of the causal depthwise conv1d. Calls
``F.conv1d`` directly without entering ``DeterministicContext``, so the
seed runs on PyTorch's fast (non-deterministic) code path.

**Expected initial reward.** Unlike TriMul (where the seed reports
~1.0x against its reference because the wrapper ``DisableCuDNNTF32``
adds near-zero overhead for ``nn.Linear``), this kernel's reference is
wrapped in ``DeterministicContext`` (``torch.use_deterministic_algorithms(True)``
+ TF32 off), which forces depthwise ``F.conv1d`` onto a slow code path
on Ampere. The seed therefore reports a baseline of roughly **2x**
against the deterministic reference. This is intentional and matches
upstream gpu-mode/reference-kernels semantics — they too score against
the deterministic reference.

**Two reward views.** The search loop optimizes
speedup-over-the-deterministic-reference (matches upstream / what an
independent reviewer would compute against the same task). For paper
reporting we additionally surface speedup-over-seed (= candidate /
seed) in the experiment results loader: that view strips out the ~2x
free determinism gain and reports only what the search actually
contributed beyond the PyTorch fast-path baseline. The seed is kept on
the fast path deliberately so the LLM cannot reward-hack by removing
``DeterministicContext`` — it isn't there to remove.

The upstream ``submission.py`` is a Helion implementation with
autotuned configs; using it as the seed would skip the cold-start
phase the search loop is designed to handle.

The ``from task import input_t, output_t`` import resolves at
candidate-load time via the synthetic ``task.py`` shim installed by
``arid_badger.causal_conv1d.scoring._loaded_candidate``.

Mirror of ``arid_badger.trimul.seed_kernel``.
"""

from __future__ import annotations


SEED_KERNEL_CODE: str = '''import torch
import torch.nn.functional as F
from task import input_t, output_t


def custom_kernel(data: input_t) -> output_t:
    """Reference causal depthwise 1D convolution in PyTorch."""
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
