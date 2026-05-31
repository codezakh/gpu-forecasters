"""Correctness oracle + numerical utilities for causal conv1d scoring.

Vendored from
``gpu-mode/reference-kernels/problems/helion/utils.py`` (``set_seed``,
``verbose_allclose``, ``match_reference``, ``make_match_reference``,
``DeterministicContext``).

Near-duplicate of ``gpu_forecasters.trimul.comparison``. The two differ in:
- this module exposes ``DeterministicContext`` (helion's name + uses
  ``torch.use_deterministic_algorithms`` + ``CUBLAS_WORKSPACE_CONFIG``)
  vs. TriMul's ``DisableCuDNNTF32`` (cuDNN flags only). The candidate
  ``reference.py`` uses ``DeterministicContext``, so the symbol name
  matters at the synthetic-utils-shim layer.
- the float32 cast in ``verbose_allclose``'s diff is taken from the
  TriMul vendor (slightly more robust against bf16 underflow); the
  textual mismatch format is also TriMul's.

The duplication is intentional pending the gh070-A task #3 extraction.
"""

from __future__ import annotations

import os
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
    """Element-wise allclose with detailed mismatch info."""
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


class DeterministicContext:
    """Context manager: disable TF32, enable deterministic cuDNN, force
    deterministic algorithms via ``torch.use_deterministic_algorithms``.

    Helion's reference kernel imports this directly; the synthetic
    ``utils`` shim re-exports it under this name so candidate sources
    that mirror upstream's ``from utils import DeterministicContext``
    resolve cleanly.
    """

    def __init__(self) -> None:
        self.allow_tf32: bool | None = None
        self.deterministic: bool | None = None
        self.cublas: str = ""

    def __enter__(self) -> "DeterministicContext":
        self.cublas = os.environ.get("CUBLAS_WORKSPACE_CONFIG", "")
        # Some kernels need a workspace setting for deterministic cublas
        # (matmul) routines on Ampere+; the helion utils don't set this
        # explicitly, leaving the default. We mirror that.
        self.allow_tf32 = torch.backends.cudnn.allow_tf32
        self.deterministic = torch.backends.cudnn.deterministic
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        if self.allow_tf32 is not None:
            torch.backends.cudnn.allow_tf32 = self.allow_tf32
        if self.deterministic is not None:
            torch.backends.cudnn.deterministic = self.deterministic
        torch.use_deterministic_algorithms(False)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = self.cublas
