"""Search-side observation types for an abstaining evaluator.

The search-embedded story (ZAI-72) needs a single observation type the
v2 search driver can carry through its event log even though the
underlying outcome is one of two very different things:

* A ``ForecastObservation`` — the surrogate forecast a runtime, and
  search proceeds without any GPU work for this candidate.
* A ``RealObservation`` — the surrogate deferred (or was never asked,
  e.g. for the bootstrap eval of the seed program), and a real
  evaluator produced a ``GpuModeKernelObservation`` the same way it
  always has.

These are not the same concept as the flat-eval ``Predict`` /
``AbstainDecision`` types in ``domain.py``: those are *policy
decisions* (what the abstainer chose), while these are *evaluator
outcomes* (what the search saw when it asked for an evaluation).
"""

from __future__ import annotations

from typing import Annotated, Generic, Literal, Union

from pydantic import BaseModel, ConfigDict, Field

from gpu_forecasters.gpu_mode_kernel.core import (
    CaseSpeedupBase,
    CaseSpeedupT,
    GpuModeKernelObservation,
)
from gpu_forecasters.landscape_map.v2 import KernelRuntimeEstimate


class ForecastObservation(BaseModel):
    """Surrogate-only outcome — no GPU run happened.

    ``expected_speedup`` is the reward the search saw for this
    candidate, materialized at observation construction time so post-
    hoc readers do not need to re-derive it from the distribution
    (and so the field that drives PUCT lives next to the events that
    record it).
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["forecast"] = "forecast"
    estimate: KernelRuntimeEstimate
    expected_speedup: float


class RealObservation(BaseModel, Generic[CaseSpeedupT]):
    """Real-eval outcome — a GPU run produced ``inner``.

    ``deferral_reason`` is populated when the surrogate produced a
    ``Deferral`` that triggered this real eval; it is ``None`` when
    the real eval bypassed the surrogate entirely (the bootstrap eval
    of the seed program).
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["real"] = "real"
    inner: GpuModeKernelObservation[CaseSpeedupT]
    deferral_reason: str | None = None


# PEP 695 generic alias — same trick as ``KernelExecutionFeedback``.
# A bare ``Annotated[Union[..., RealObservation[CaseSpeedupT]], ...]``
# would capture ``CaseSpeedupT`` at module-load time and could not be
# rebound at use sites; the ``type X[T] = ...`` form produces a
# ``TypeAliasType`` that Pydantic v2 resolves through subscription
# when constructing TypeAdapters / generic models.
type CompoundObservation[T: CaseSpeedupBase] = Annotated[
    Union[ForecastObservation, RealObservation[T]],
    Field(discriminator="kind"),
]


__all__ = [
    "CompoundObservation",
    "ForecastObservation",
    "RealObservation",
]
