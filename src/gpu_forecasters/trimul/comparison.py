"""Correctness oracle + numerical utilities for TriMul scoring.

Vendored from ``ttt-discover/examples/gpu_mode/lib/bioml/trimul/utils.py``
(functions: ``set_seed``, ``verbose_allclose``, ``match_reference``,
``make_match_reference``, ``DisableCuDNNTF32``). Copies, not imports —
see design note #2 in the port plan (ttt-discover is an external checkout
and its bare ``from task import ...`` imports do not resolve inside our
library).
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
    output: torch.Tensor,
    reference: Callable[[Any], torch.Tensor],
    rtol: float = 1e-05,
    atol: float = 1e-08,
) -> Tuple[bool, str]:
    """Default ``check_implementation`` implementation used by tasks.

    Runs the reference and compares against ``output``. Returns
    ``(good, message)``.
    """
    expected = reference(data)
    good, reasons = verbose_allclose(output, expected, rtol=rtol, atol=atol)
    if len(reasons) > 0:
        return good, "\n".join(reasons)
    return good, ""


def make_match_reference(
    reference: Callable[[Any], torch.Tensor], **kwargs: Any
) -> Callable[[Any, torch.Tensor], Tuple[bool, str]]:
    def wrapped(data: Any, output: torch.Tensor) -> Tuple[bool, str]:
        return match_reference(data, output, reference=reference, **kwargs)

    return wrapped


class DisableCuDNNTF32:
    """Context manager: disable TF32 and enable deterministic cuDNN.

    Wraps the reference pass so that correctness checks are stable across
    runs and GPU architectures.
    """

    def __init__(self) -> None:
        self.allow_tf32 = torch.backends.cudnn.allow_tf32
        self.deterministic = torch.backends.cudnn.deterministic

    def __enter__(self) -> "DisableCuDNNTF32":
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        torch.backends.cudnn.allow_tf32 = self.allow_tf32
        torch.backends.cudnn.deterministic = self.deterministic
