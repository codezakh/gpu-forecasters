"""Discriminated-union outcome types for the abstain-or-forecast surrogate.

The model is given two tools — predict and defer — and must call
exactly one. Whatever it calls, the parser yields one of two values:

* :class:`Forecast` wraps the existing :class:`KernelRuntimeEstimate`.
* :class:`Deferral` carries the LLM's deferral rationale.

These types are infrastructure-free (no LiteLLM, no Tinker, no
cookbook). Both estimator paths (LiteLLM-backed eval, Tinker-backed
training and trained-checkpoint scoring) parse their respective
upstream responses into one of these two arms.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from gpu_forecasters.landscape_map.v2.domain import KernelRuntimeEstimate


class Deferral(BaseModel):
    """Domain object for a native-abstain decision.

    Carries the LLM's deferral rationale; downstream code reading the
    log can group abstentions by reason for diagnostics.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["defer"] = "defer"
    reason: str


class Forecast(BaseModel):
    """Domain object for a native-predict decision.

    Wraps the existing :class:`KernelRuntimeEstimate` so the union
    discriminator can sit on ``kind``.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["predict"] = "predict"
    estimate: KernelRuntimeEstimate


PredictOrDefer = Annotated[
    Union[Forecast, Deferral],
    Field(discriminator="kind"),
]


__all__ = [
    "Deferral",
    "Forecast",
    "PredictOrDefer",
]
