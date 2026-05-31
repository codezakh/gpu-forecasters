"""Events for the theory-builder outer loop.

Modeled directly on ``max_reward_puct.v2.events`` — every async step
splits into ``Requested``/``Completed``/``Failed`` so a mid-step crash
plus restart is recoverable from the log alone.

The ``WorldModel`` is a fold of this log; it is never persisted
independently. Three async steps drive the loop:

* **Hypothesis.** The builder reads the current world model and
  proposes a new ``Hypothesis``.
* **Experiment.** The worker runs an inner search seeded by the
  hypothesis and returns an ``ExperimentResult``.
* **Explanation.** The builder reads the result and emits an
  ``Explanation`` plus diffs to apply to the world model.

A fourth fence event, ``OuterStepCompleted``, folds the explanation's
diffs into the world model text and bumps the step counter.

Generic over ``ObservationT`` (e.g. ``TriMulObservation``). The
discriminated-union TypeAdapter is built per-instance via
``theory_event_adapter`` — Python's typing machinery cannot
re-parameterize an already-parameterized ``Annotated[Union, ...]``.
"""

from __future__ import annotations

from typing import Annotated, Generic, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from gpu_forecasters.hill_climbing.domain import ObservationT
from gpu_forecasters.theory_builder.v1.domain import (
    Explanation,
    ExperimentResult,
    Hypothesis,
    WorldModel,
)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True)


class TheoryBuildingInitialized(_Frozen):
    """Bootstraps the loop with the kernel description and the empty
    world model. Always the first event in a log."""

    kind: Literal["theory_building_initialized"] = "theory_building_initialized"
    world_model: WorldModel


class HypothesisRequested(_Frozen):
    kind: Literal["hypothesis_requested"] = "hypothesis_requested"
    request_id: str


class HypothesisCompleted(_Frozen):
    kind: Literal["hypothesis_completed"] = "hypothesis_completed"
    request_id: str
    hypothesis: Hypothesis


class HypothesisFailed(_Frozen):
    kind: Literal["hypothesis_failed"] = "hypothesis_failed"
    request_id: str
    reason: str


class ExperimentRequested(_Frozen):
    """Carries the hypothesis directly so re-dispatch on recovery
    needs no other context."""

    kind: Literal["experiment_requested"] = "experiment_requested"
    request_id: str
    hypothesis: Hypothesis


class ExperimentCompleted(_Frozen, Generic[ObservationT]):
    kind: Literal["experiment_completed"] = "experiment_completed"
    request_id: str
    result: ExperimentResult[ObservationT]


class ExperimentFailed(_Frozen):
    kind: Literal["experiment_failed"] = "experiment_failed"
    request_id: str
    reason: str


class ExplanationRequested(_Frozen, Generic[ObservationT]):
    """Carries the hypothesis + result so re-dispatch on recovery
    needs no other context."""

    kind: Literal["explanation_requested"] = "explanation_requested"
    request_id: str
    hypothesis: Hypothesis
    result: ExperimentResult[ObservationT]


class ExplanationCompleted(_Frozen):
    kind: Literal["explanation_completed"] = "explanation_completed"
    request_id: str
    explanation: Explanation
    # The post-apply world-model text. We log it explicitly so replay
    # is deterministic even if the diff applier's behaviour changes.
    new_world_model_text: str


class ExplanationFailed(_Frozen):
    kind: Literal["explanation_failed"] = "explanation_failed"
    request_id: str
    reason: str


class OuterStepCompleted(_Frozen):
    """Fence event. The reducer uses this to bump the outer step
    counter. The world-model text was already updated in
    ``ExplanationCompleted`` (so the world model is consistent inside
    the same step)."""

    kind: Literal["outer_step_completed"] = "outer_step_completed"
    step: int


TheoryEvent = Annotated[
    Union[
        TheoryBuildingInitialized,
        HypothesisRequested,
        HypothesisCompleted,
        HypothesisFailed,
        ExperimentRequested,
        ExperimentCompleted[ObservationT],
        ExperimentFailed,
        ExplanationRequested[ObservationT],
        ExplanationCompleted,
        ExplanationFailed,
        OuterStepCompleted,
    ],
    Field(discriminator="kind"),
]
"""Generic discriminated union — for type annotations only. Build a
runtime ``TypeAdapter`` via ``theory_event_adapter`` (concrete
``ObservationT``)."""


def theory_event_adapter(
    observation_type: type[ObservationT],
) -> TypeAdapter[TheoryEvent[ObservationT]]:
    """Build a ``TypeAdapter`` specialized on a concrete ``ObservationT``."""
    union = Annotated[
        Union[
            TheoryBuildingInitialized,
            HypothesisRequested,
            HypothesisCompleted,
            HypothesisFailed,
            ExperimentRequested,
            ExperimentCompleted[observation_type],  # type: ignore[valid-type]
            ExperimentFailed,
            ExplanationRequested[observation_type],  # type: ignore[valid-type]
            ExplanationCompleted,
            ExplanationFailed,
            OuterStepCompleted,
        ],
        Field(discriminator="kind"),
    ]
    return TypeAdapter(union)


__all__ = [
    "TheoryEvent",
    "TheoryBuildingInitialized",
    "HypothesisRequested",
    "HypothesisCompleted",
    "HypothesisFailed",
    "ExperimentRequested",
    "ExperimentCompleted",
    "ExperimentFailed",
    "ExplanationRequested",
    "ExplanationCompleted",
    "ExplanationFailed",
    "OuterStepCompleted",
    "theory_event_adapter",
]
