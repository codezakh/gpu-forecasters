"""Generic execution feedback types for gpu-mode-style scoring.

This module is the kernel-agnostic core of ``gpu_forecasters.gpu_mode_kernel``.
It generalizes the per-kernel ``core.py`` modules under
``gpu_forecasters.trimul`` and ``gpu_forecasters.causal_conv1d``:

- The four failure feedback arms (``CompileFailedFeedback``,
  ``RuntimeErrorFeedback``, ``IncorrectFeedback``,
  ``InfrastructureFailureFeedback``) are wholly kernel-agnostic and lift
  verbatim.
- ``Stats`` (timing summary) and ``KernelExecResult`` (the wire-format
  return type of ``score_one_case``) are also kernel-agnostic — neither
  carries shape fields, so they work for every kernel.
- ``CaseSpeedup`` is the only kernel-specific shape (B/D/S/W vs
  seqlen/bs/dim/hiddendim vs B/V…). Concrete packs subclass
  ``CaseSpeedupBase`` to add their shape fields.
- ``SuccessFeedback`` is generic over the case-speedup type. The
  discriminated union ``KernelExecutionFeedback[CaseSpeedupT]`` is
  parameterized at the pack's use site.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Any, Generic, Literal, Mapping, Self, TypeVar, Union

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Adaptive-loop stats (lifted byte-identical from per-kernel cores).
# ---------------------------------------------------------------------------


@dataclass
class Stats:
    """Timing statistics from an adaptive benchmarking loop."""

    runs: int
    mean: float
    std: float
    err: float
    best: float
    worst: float


# ---------------------------------------------------------------------------
# CaseSpeedup base class — concrete packs subclass with shape fields.
# ---------------------------------------------------------------------------


class CaseSpeedupBase(BaseModel):
    """Base class for per-case speedup records.

    Concrete kernel packs subclass and add the shape fields that
    parameterize the test case (``B``, ``D``, ``S``, ``W`` for
    causal_conv1d; ``seqlen``, ``bs``, ``dim``, ``hiddendim``,
    ``nomask``, ``distribution`` for trimul; ``vocab_size`` for
    cross_entropy).

    The three timing fields below are always present regardless of
    kernel. Subclasses must override ``from_exec_result`` to populate
    their shape fields, and ``format_for_prompt`` to produce the
    per-case line shown to the LLM in the mutation feedback prompt.
    """

    model_config = ConfigDict(frozen=True)

    speedup: float
    runtime_ns: float
    ref_runtime_ns: float

    @classmethod
    def from_exec_result(
        cls,
        test_args: Mapping[str, Any],
        exec_result: "KernelExecResult",
    ) -> Self:
        """Build a ``CaseSpeedup`` from the test args + exec result.

        Concrete subclasses must override to read shape fields out of
        ``test_args`` and populate them on ``cls(...)``. Raising rather
        than providing a default so a missing override fails loudly at
        first call instead of silently producing a CaseSpeedup with
        empty shape data.
        """
        raise NotImplementedError(
            f"{cls.__name__} must override CaseSpeedupBase.from_exec_result "
            "to read its shape fields from test_args."
        )

    def format_for_prompt(self) -> str:
        """Format this case for the mutation feedback prompt.

        Concrete subclasses must override to include their shape
        fields in the per-case line shown to the LLM.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must override CaseSpeedupBase."
            "format_for_prompt to render its shape fields."
        )


CaseSpeedupT = TypeVar("CaseSpeedupT", bound=CaseSpeedupBase)


# ---------------------------------------------------------------------------
# Feedback union — three failure arms + one generic success arm.
# ---------------------------------------------------------------------------


class CompileFailedFeedback(BaseModel):
    """Module import / syntax error.

    Candidates are plain Python modules exposing ``custom_kernel``; a
    ``SyntaxError`` or ``NameError`` during module import stands in for
    nvcc's compile failure.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["compile_failed"] = "compile_failed"
    compilation_error: str


class RuntimeErrorFeedback(BaseModel):
    """Candidate's ``custom_kernel`` raised at runtime."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["runtime_error"] = "runtime_error"
    runtime_error_name: str
    runtime_error: str
    traceback: str


class IncorrectFeedback(BaseModel):
    """Candidate ran but produced output outside tolerance of reference."""

    model_config = ConfigDict(frozen=True)

    kind: Literal["incorrect"] = "incorrect"
    error_message: str


class SuccessFeedback(BaseModel, Generic[CaseSpeedupT]):
    """Candidate passed correctness on all cases; aggregated timings.

    Generic over the per-case speedup shape supplied by the kernel
    pack. The provider layer constructs this with the pack's concrete
    ``CaseSpeedup`` subclass so that downstream consumers (mutation
    prompts, plots, event-log readers) can read shape fields.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["success"] = "success"
    aggregated_speedup: float
    aggregation_method: str
    per_case_speedups: list[CaseSpeedupT]


class InfrastructureFailureFeedback(BaseModel):
    """Scoring harness crashed before producing a verdict.

    Distinct from the in-union failure arms: this signals that we
    couldn't determine whether the candidate is correct or not (Modal
    crash, bad GPU state). Not part of ``KernelExecutionFeedback`` —
    callers handle this case explicitly because it cannot drive the
    mutation prompt's "here's why your kernel failed" feedback path.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["infrastructure_failure"] = "infrastructure_failure"
    reason: str


# Discriminated union over the in-band outcomes (i.e. ones the candidate
# itself produces). ``InfrastructureFailureFeedback`` is intentionally
# excluded — it represents a harness fault rather than a candidate
# outcome.
#
# PEP 695 generic type alias (Python 3.12+): ``type X[T] = ...`` produces
# a ``TypeAliasType`` that is subscriptable at runtime, unlike a bare
# ``Annotated[Union[..., SuccessFeedback[CaseSpeedupT]], ...]`` whose
# ``CaseSpeedupT`` is captured at module-load time and cannot be
# rebound. Pydantic v2 resolves ``TypeAliasType`` through subscription
# when constructing TypeAdapters / generic models.
type KernelExecutionFeedback[T: CaseSpeedupBase] = Annotated[
    Union[
        CompileFailedFeedback,
        RuntimeErrorFeedback,
        IncorrectFeedback,
        SuccessFeedback[T],
    ],
    Field(discriminator="kind"),
]


# 5-arm superset that adds ``InfrastructureFailureFeedback`` for the
# observation/aggregation surfaces — search needs to distinguish "your
# candidate did something" from "the harness crashed."
type ObservationFeedback[T: CaseSpeedupBase] = Annotated[
    Union[
        CompileFailedFeedback,
        RuntimeErrorFeedback,
        IncorrectFeedback,
        SuccessFeedback[T],
        InfrastructureFailureFeedback,
    ],
    Field(discriminator="kind"),
]


# ---------------------------------------------------------------------------
# KernelExecResult — wire-format per-case raw outcome from ``score_one_case``.
# ---------------------------------------------------------------------------


class KernelExecResult(BaseModel):
    """Raw outcome of running one candidate on one test case.

    ``score_one_case`` returns ``Ok(KernelExecResult)`` on successful
    execution of the scoring pipeline — even if the candidate itself
    was wrong. Only infrastructure failures (Modal crash, etc.)
    produce ``Err``. The provider layer turns a list of these into a
    ``KernelExecutionFeedback`` (via
    ``failure_feedback_from_exec_result`` for failures, or by
    constructing ``SuccessFeedback`` directly for the all-correct
    path).

    Kernel-agnostic: shape fields live on ``CaseSpeedup`` (which is
    built by the provider by ``zip``-ing ``test_cases`` with these
    exec results), not here.

    Why this is a flat record rather than a discriminated union: it
    is the wire format crossing the Modal cls boundary. Modal
    serializes return values via cloudpickle, and a flat Pydantic
    model with a single ``failure_kind`` discriminator-string round-
    trips reliably across the boundary in a way that a generic
    discriminated union over ``SuccessFeedback[CaseSpeedupT]`` plus
    failure arms does not. The provider re-discriminates into
    ``KernelExecutionFeedback`` on this side via
    ``failure_feedback_from_exec_result``.
    """

    model_config = ConfigDict(frozen=True)

    correct: bool
    runtime_ns: float
    ref_runtime_ns: float
    error_message: str = ""
    runtime_error_name: str = ""
    runtime_error: str = ""
    traceback: str = ""
    compilation_error: str = ""
    failure_kind: Literal[
        "none", "compile_failed", "runtime_error", "incorrect"
    ] = "none"


KernelFailureFeedback = Union[
    CompileFailedFeedback,
    RuntimeErrorFeedback,
    IncorrectFeedback,
]


def failure_feedback_from_exec_result(
    exec_result: KernelExecResult,
) -> KernelFailureFeedback:
    """Build a failure feedback from a failing ``KernelExecResult``.

    Only valid when ``exec_result.failure_kind != "none"``. The
    all-correct path constructs ``SuccessFeedback`` directly in the
    provider layer (which has the multi-case context needed for
    aggregation).
    """
    match exec_result.failure_kind:
        case "compile_failed":
            return CompileFailedFeedback(
                compilation_error=exec_result.compilation_error,
            )
        case "runtime_error":
            return RuntimeErrorFeedback(
                runtime_error_name=exec_result.runtime_error_name,
                runtime_error=exec_result.runtime_error,
                traceback=exec_result.traceback,
            )
        case "incorrect":
            return IncorrectFeedback(error_message=exec_result.error_message)
        case "none":
            raise ValueError(
                "failure_feedback_from_exec_result called on a passing result "
                "(failure_kind='none'). Use SuccessFeedback directly for the "
                "all-correct path."
            )


# ---------------------------------------------------------------------------
# Observation — what the EvaluationProvider hands to search.
# ---------------------------------------------------------------------------


class GpuModeKernelObservation(BaseModel, Generic[CaseSpeedupT]):
    """Per-candidate observation for a gpu-mode-style kernel evaluation.

    Carries either an in-band feedback (one of the
    ``KernelExecutionFeedback`` arms — used by the mutation prompt) or
    an ``InfrastructureFailureFeedback`` (signals that the search
    should not feed this back to the LLM as if the candidate
    misbehaved). ``per_case_results`` is the raw wire data, useful for
    post-hoc analysis even when the feedback summary discards detail.
    """

    model_config = ConfigDict(frozen=True)

    feedback: ObservationFeedback[CaseSpeedupT]
    per_case_results: list[KernelExecResult] = []
