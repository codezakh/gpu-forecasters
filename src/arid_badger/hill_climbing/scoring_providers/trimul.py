"""Observation type for TriMul scoring.

Parallels ``scoring_providers/kernelbench.py``: a single-field wrapper
around the TriMul-specific feedback union, so that ``EvaluationProvider``
can be parameterised by a concrete ``BaseModel`` subclass rather than a
bare ``Union``.
"""

from __future__ import annotations

from typing import Annotated, Union

from pydantic import BaseModel, ConfigDict, Field

from arid_badger.trimul.core import (
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

    feedback: TriMulFeedback
    per_case_results: list[TriMulExecResult] = []
