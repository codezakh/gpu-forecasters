"""Modal GPU identifiers, shared across providers.

``GpuKind`` is a ``StrEnum`` whose values match Modal's ``gpu=`` argument
exactly so members pass through ``.with_options(gpu=...)`` unchanged. Use
this as the canonical type for any provider-config or constructor field
that selects a Modal GPU; it gives type-checked typo detection at the
call boundary while remaining string-compatible at lookup sites (e.g.
the ``COMPUTE_CAPABILITY_BY_GPU`` and ``GPU_ARCH_MAPPING`` dicts in
``gpu_forecasters.kernelbench``).

Members enumerate the GPUs the kernelbench / gpu_mode_kernel infra has
arch + compute-capability mappings for today. Add a new member here and
extend the relevant infra dict together — the enum is the source of
truth for "GPUs this codebase supports."
"""

from __future__ import annotations

from enum import StrEnum


class GpuKind(StrEnum):
    """Modal GPU identifiers. String values match Modal's ``gpu=`` argument
    exactly so they pass through ``.with_options(gpu=...)`` unchanged."""

    T4 = "T4"
    A10G = "A10G"
    A100 = "A100"
    A100_80GB = "A100-80GB"
    L4 = "L4"
    L40S = "L40S"
    H100 = "H100"
    H200 = "H200"
    B200 = "B200"
