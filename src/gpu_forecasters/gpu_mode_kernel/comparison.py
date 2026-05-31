"""Kernel-agnostic correctness oracle + numerical utilities.

Lifted from ``gpu_forecasters.trimul.comparison`` and
``gpu_forecasters.causal_conv1d.comparison`` — these helpers were byte-identical
across both packages. Per-kernel determinism context managers
(``DisableCuDNNTF32``, ``DeterministicContext``, …) live on the
``KernelPack`` rather than here, because their identity (the symbol name a
candidate's ``utils.py`` shim re-exports) is kernel-specific.

Vendored origin: ``ttt-discover/examples/gpu_mode/lib/bioml/trimul/utils.py``
(``set_seed``, ``verbose_allclose``, ``match_reference``,
``make_match_reference``).
"""

from __future__ import annotations

import random
from typing import Any, Callable, Tuple

import numpy as np
import torch


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def verbose_allclose(
    received: torch.Tensor,
    expected: torch.Tensor,
    rtol: float = 1e-05,
    atol: float = 1e-08,
    max_print: int = 5,
) -> Tuple[bool, list[str]]:
    """Element-wise allclose with detailed mismatch info.

    Adapted from Liger-Kernel's test/utils.py via ttt-discover's
    ``trimul/utils.py``.
    """
    if received.shape != expected.shape:
        return False, ["SIZE MISMATCH"]

    diff = torch.abs(received.to(torch.float32) - expected.to(torch.float32))
    tolerance = atol + rtol * torch.abs(expected)

    tol_mismatched = diff > tolerance
    nan_mismatched = torch.logical_xor(torch.isnan(received), torch.isnan(expected))
    posinf_mismatched = torch.logical_xor(
        torch.isposinf(received), torch.isposinf(expected)
    )
    neginf_mismatched = torch.logical_xor(
        torch.isneginf(received), torch.isneginf(expected)
    )

    mismatched = torch.logical_or(
        torch.logical_or(tol_mismatched, nan_mismatched),
        torch.logical_or(posinf_mismatched, neginf_mismatched),
    )

    mismatched_indices = torch.nonzero(mismatched)
    num_mismatched = int(mismatched.count_nonzero().item())

    if num_mismatched >= 1:
        details = [f"Number of mismatched elements: {num_mismatched}"]
        for index in mismatched_indices[:max_print]:
            i = tuple(index.tolist())
            details.append(f"ERROR at {i}: {received[i]} {expected[i]}")
        if num_mismatched > max_print:
            details.append(
                f"... and {num_mismatched - max_print} more mismatched elements."
            )
        return False, details

    return True, [f"Maximum error: {torch.max(diff)}"]


def match_reference(
    data: Any,
    output: Any,
    reference: Callable[[Any], Any],
    rtol: float = 1e-05,
    atol: float = 1e-08,
) -> Tuple[bool, str]:
    """Default ``check_implementation`` implementation.

    Runs the reference and compares against ``output``. Returns
    ``(good, message)``. ``output`` and the reference's return type need
    not be a single tensor — kernels with multi-output references (e.g.
    cross-entropy fwd+bwd returning ``(loss, grad)``) override
    ``check_implementation`` on their ``KernelPack`` rather than going
    through this helper.
    """
    expected = reference(data)
    good, reasons = verbose_allclose(output, expected, rtol=rtol, atol=atol)
    if len(reasons) > 0:
        return good, "\n".join(reasons)
    return good, ""


def make_match_reference(
    reference: Callable[[Any], Any], **kwargs: Any
) -> Callable[[Any, Any], Tuple[bool, str]]:
    def wrapped(data: Any, output: Any) -> Tuple[bool, str]:
        return match_reference(data, output, reference=reference, **kwargs)

    return wrapped
