"""The v2-local outcome union for a single rollout.

Composes the four ``TriMulKernelExecutionFeedback`` variants from
``gpu_forecasters.trimul.core`` with two additional variants that v2
distinguishes explicitly: ``InfrastructureFailureFeedback`` (Modal /
scoring harness crash, separable from candidate-level failures for
reward / analysis purposes) and ``ParseFailureFeedback`` (the LLM's
response did not contain an extractable code block — cannot proceed to
evaluation, not the evaluator's fault).
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from gpu_forecasters.trimul.core import (
    CompileFailedFeedback,
    IncorrectFeedback,
    InfrastructureFailureFeedback,
    RuntimeErrorFeedback,
    SuccessFeedback,
)


class ParseFailureFeedback(BaseModel):
    """The model response contained no extractable code block.

    Distinct from ``CompileFailedFeedback`` (which implies the extracted
    code failed to import) and ``InfrastructureFailureFeedback`` (Modal
    crash). Surfaces as its own variant so that a policy that e.g. always
    produces text without a trailing code block can be identified in the
    event log without grepping reason strings.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["parse_failure"] = "parse_failure"
    reason: str


TriMulRLOutcome = Annotated[
    Union[
        CompileFailedFeedback,
        RuntimeErrorFeedback,
        IncorrectFeedback,
        SuccessFeedback,
        InfrastructureFailureFeedback,
        ParseFailureFeedback,
    ],
    Field(discriminator="kind"),
]
