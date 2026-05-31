"""Observation type for causal conv1d scoring.

Mirror of ``scoring_providers/trimul`` — a single-field wrapper around
the kernel-specific feedback union, so that ``EvaluationProvider`` can
be parameterised by a concrete ``BaseModel`` subclass rather than a
bare ``Union``.
"""

from __future__ import annotations

from typing import Annotated, Union

from pydantic import BaseModel, ConfigDict, Field

from gpu_forecasters.causal_conv1d.core import (
    CausalConv1dExecResult,
    CausalConv1dKernelExecutionFeedback,
    InfrastructureFailureFeedback,
)


CausalConv1dFeedback = Annotated[
    Union[CausalConv1dKernelExecutionFeedback, InfrastructureFailureFeedback],
    Field(discriminator="kind"),
]


class CausalConv1dObservation(BaseModel):
    """Observation from a causal conv1d scoring attempt."""

    model_config = ConfigDict(frozen=True)

    feedback: CausalConv1dFeedback
    per_case_results: list[CausalConv1dExecResult] = []
