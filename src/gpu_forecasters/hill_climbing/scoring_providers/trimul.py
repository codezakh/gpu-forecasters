"""Observation type for TriMul scoring.

Parallels ``scoring_providers/kernelbench.py``: a single-field wrapper
around the TriMul-specific feedback union, so that ``EvaluationProvider``
can be parameterised by a concrete ``BaseModel`` subclass rather than a
bare ``Union``.
"""

from __future__ import annotations

from typing import Annotated, Union

from pydantic import BaseModel, ConfigDict, Field

from gpu_forecasters.trimul.core import (
    InfrastructureFailureFeedback,
    TriMulExecResult,
    TriMulKernelExecutionFeedback,
)


TriMulFeedback = Annotated[
    Union[TriMulKernelExecutionFeedback, InfrastructureFailureFeedback],
    Field(discriminator="kind"),
]


class TriMulObservation(BaseModel):
    """Observation from a TriMul scoring attempt."""

    model_config = ConfigDict(frozen=True)

    # Structured summary used for reward computation and mutation prompts.
    feedback: TriMulFeedback
    # Raw per-case wire-format results for downstream inspection.  Overlaps
    # with SuccessFeedback.per_case_speedups on the success path, but carries
    # the full TriMulExecResult (error messages, tracebacks, etc.) which the
    # feedback summary intentionally does not.
    per_case_results: list[TriMulExecResult] = []
