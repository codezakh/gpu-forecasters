"""``KernelPack`` — value object that parameterizes generic gpu-mode infra.

The seam: every kernel-specific surface (reference, cases, seed, prompt
body, candidate-load shims) collapses into one frozen dataclass that the
generic scoring + Modal + provider layers consume. Adding a new
gpu-mode/reference-kernels problem becomes a single-file declaration
under ``gpu_forecasters.gpu_mode_kernel.packs.<name>`` rather than a ~10-file
mirror of ``gpu_forecasters.trimul`` / ``gpu_forecasters.causal_conv1d``.

Type parameters:
- ``TestArgsT``: TypedDict describing one test case's inputs (e.g.
  ``CausalConv1dTestArgs``). Opaque to infra; threaded into
  ``generate_input``.
- ``CaseSpeedupT``: per-case speedup record subclass of
  ``CaseSpeedupBase``. The pack's ``case_speedup_factory`` builds these
  on the success path; the prompt formatter reads their shape fields
  out.
"""

from __future__ import annotations

import sys
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Callable, Generic, Tuple, Type, TypeVar

from gpu_forecasters.gpu_mode_kernel.core import CaseSpeedupBase


TestArgsT = TypeVar("TestArgsT")
CaseSpeedupT = TypeVar("CaseSpeedupT", bound=CaseSpeedupBase)


@dataclass(frozen=True)
class KernelPack(Generic[TestArgsT, CaseSpeedupT]):
    """All kernel-specific behavior, in one value.

    Construct one of these per kernel under
    ``gpu_forecasters.gpu_mode_kernel.packs.<kernel>`` and export it as a
    module-level constant (e.g. ``CROSS_ENTROPY_PACK``). Generic infra
    consumes it without further parameterization.
    """

    # --- Identity ------------------------------------------------------

    name: str
    """Short kernel name, e.g. ``"trimul"``, ``"causal_conv1d"``."""

    modal_app_name: str
    """Modal app namespace, e.g. ``"arid-badger-trimul"``. Each kernel
    gets its own app so container caches and ``single_use_containers``
    semantics don't cross-contaminate."""

    # --- Test cases ----------------------------------------------------

    correctness_cases: list[TestArgsT]
    """Cases used for the obligatory correctness pass. Smaller shapes."""

    benchmark_cases: list[TestArgsT]
    """Cases used for adaptive timing on the success path."""

    # --- Reference behavior -------------------------------------------

    ref_kernel: Callable[[Any], Any]
    """Reference implementation. Output type matches the candidate's;
    may be a tensor, a tuple, etc. Must be wrapped in the kernel's
    determinism context manager so its timing is reproducible."""

    generate_input: Callable[..., Any]
    """``generate_input(**test_args) -> data`` — produces a fresh input
    on the GPU for one test case. Called once per case per scoring run;
    a fresh copy is cloned for each timed iteration."""

    check_implementation: Callable[[Any, Any], Tuple[bool, str]]
    """``check_implementation(data, candidate_output) -> (good, msg)``.
    Most packs build this via
    ``gpu_forecasters.gpu_mode_kernel.comparison.make_match_reference``;
    multi-output kernels (cross-entropy fwd+bwd) supply a custom
    callable."""

    # --- Seed kernel ---------------------------------------------------

    seed_kernel_code: str
    """Source of the cold-start ``custom_kernel`` candidate. A trivial
    PyTorch wrapper for most kernels."""

    # --- Candidate loading: determinism context manager --------------

    determinism_ctx: Type[AbstractContextManager[Any]] | None
    """The kernel's determinism context manager (e.g.
    ``DisableCuDNNTF32`` for TriMul, ``DeterministicContext`` for
    causal_conv1d, ``None`` for cross_entropy).

    The candidate's ``utils.py`` shim re-exports this so upstream
    submission templates resolve their ``from utils import …`` line
    without modification. The import line is generated from
    ``cls.__module__`` and ``cls.__name__``, so the class must live
    at a stable, importable path (i.e. a module attribute, not a
    closure).

    The kernel-agnostic helpers (``set_seed``, ``verbose_allclose``,
    ``match_reference``, ``make_match_reference``) are always
    re-exported from ``gpu_forecasters.gpu_mode_kernel.comparison`` —
    no per-pack configuration needed."""

    # --- Per-case record ----------------------------------------------

    case_speedup_type: Type[CaseSpeedupT]
    """The pack's concrete ``CaseSpeedup`` Pydantic subclass.

    Used to (a) parameterize ``SuccessFeedback`` /
    ``GpuModeKernelObservation`` in the provider, and (b) build per-
    case records via ``cls.from_exec_result(test_args, exec_result)``.
    The subclass also provides ``format_for_prompt(self) -> str`` for
    the mutation feedback prompt's per-case lines.
    """

    # --- Prompt --------------------------------------------------------

    kernel_description_body: str
    """Kernel-specific narrative for the mutation prompt: describes the
    op, lists shape parameters, includes reference source. Generic
    scaffolding (rules block, GPU/Triton block, parent-code block,
    per-case-feedback block) is added by
    ``gpu_forecasters.gpu_mode_kernel.prompts``."""

    def __post_init__(self) -> None:
        # Fail fast if ``determinism_ctx`` was assigned a class whose
        # ``__module__``/``__name__`` won't round-trip through an
        # ``import``. The candidate-loading shim writes
        # ``f"from {cls.__module__} import {cls.__name__}"`` into
        # ``utils.py``; without this check, a mistake (e.g. a class
        # defined inside a function) surfaces as a confusing
        # ``ImportError`` at scoring time rather than at pack
        # construction.
        ctx = self.determinism_ctx
        if ctx is None:
            return
        module = sys.modules.get(ctx.__module__)
        if module is None or getattr(module, ctx.__name__, None) is not ctx:
            raise ValueError(
                f"determinism_ctx={ctx!r} is not importable as "
                f"`from {ctx.__module__} import {ctx.__name__}` — the "
                f"candidate-loading shim cannot reference it. Define "
                f"the context manager as a module-level class."
            )
