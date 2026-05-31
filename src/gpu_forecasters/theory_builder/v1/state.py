"""Pure state + reducer for the theory-builder loop.

``TheoryState`` is whatever you get by folding a log of ``TheoryEvent``s
through ``apply_event``. No I/O, no LLM, no Modal — just data.

Inside-step bookkeeping mirrors the v2 PUCT reducer in spirit but is
much simpler: at most one hypothesis / experiment / explanation is
in flight per outer step, so all state lives in three optional
``InFlight*`` slots rather than dicts.

Recovery rules (used by the driver):

* ``in_flight_hypothesis`` set, ``hypothesis`` unset → re-dispatch
  the propose call.
* ``hypothesis`` set, ``in_flight_experiment`` set, ``result`` unset
  → re-dispatch the worker.
* ``hypothesis`` set, ``result`` set, ``in_flight_explanation`` set
  → re-dispatch the explain call.
* ``hypothesis`` set, ``result`` set, ``explanation`` set,
  ``OuterStepCompleted`` not yet fired → emit ``OuterStepCompleted``
  to fence the step.
"""

from __future__ import annotations

from typing import Generic, Optional, Sequence

from pydantic import BaseModel, ConfigDict, Field

from gpu_forecasters.hill_climbing.domain import ObservationT
from gpu_forecasters.theory_builder.v1.domain import (
    Explanation,
    ExperimentResult,
    Hypothesis,
    WorldModel,
)
from gpu_forecasters.theory_builder.v1.events import (
    ExperimentCompleted,
    ExperimentFailed,
    ExperimentRequested,
    ExplanationCompleted,
    ExplanationFailed,
    ExplanationRequested,
    HypothesisCompleted,
    HypothesisFailed,
    HypothesisRequested,
    OuterStepCompleted,
    TheoryBuildingInitialized,
    TheoryEvent,
)


class TheoryState(BaseModel, Generic[ObservationT]):
    """Everything needed to continue an outer-loop run."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    world_model: WorldModel = Field(
        default_factory=lambda: WorldModel(kernel_description="")
    )

    # All completed hypotheses, in order. Used for prompt context and
    # for inspection / debugging. Status reflects the latest known
    # value the LLM has written into the world model — but the loop
    # doesn't enforce a separate typed status field; the in-text
    # status (in ``world_model.text``) is authoritative.
    completed_hypotheses: list[Hypothesis] = Field(default_factory=list)

    # Number of outer steps *completed*. 0 means no
    # ``OuterStepCompleted`` has been seen yet.
    current_step: int = 0

    # --- Within-step bookkeeping. All cleared on OuterStepCompleted. ---

    # Hypothesis request currently in flight (or completed within
    # this step but not yet superseded by an explanation/fence).
    in_flight_hypothesis_request_id: Optional[str] = None
    hypothesis: Optional[Hypothesis] = None

    in_flight_experiment_request_id: Optional[str] = None
    result: Optional[ExperimentResult[ObservationT]] = None

    in_flight_explanation_request_id: Optional[str] = None
    explanation: Optional[Explanation] = None


def apply_event(
    state: TheoryState[ObservationT],
    event: TheoryEvent[ObservationT],
) -> TheoryState[ObservationT]:
    """Fold one event into ``state``. State is mutated in place as a
    perf concession; the result depends only on the inputs."""
    match event:
        case TheoryBuildingInitialized():
            state.world_model = event.world_model

        case HypothesisRequested():
            state.in_flight_hypothesis_request_id = event.request_id
            state.hypothesis = None

        case HypothesisCompleted():
            if state.in_flight_hypothesis_request_id == event.request_id:
                state.in_flight_hypothesis_request_id = None
            state.hypothesis = event.hypothesis

        case HypothesisFailed():
            if state.in_flight_hypothesis_request_id == event.request_id:
                state.in_flight_hypothesis_request_id = None

        case ExperimentRequested():
            state.in_flight_experiment_request_id = event.request_id
            state.result = None

        case ExperimentCompleted():
            if state.in_flight_experiment_request_id == event.request_id:
                state.in_flight_experiment_request_id = None
            state.result = event.result

        case ExperimentFailed():
            if state.in_flight_experiment_request_id == event.request_id:
                state.in_flight_experiment_request_id = None

        case ExplanationRequested():
            state.in_flight_explanation_request_id = event.request_id
            state.explanation = None

        case ExplanationCompleted():
            if state.in_flight_explanation_request_id == event.request_id:
                state.in_flight_explanation_request_id = None
            state.explanation = event.explanation
            # Trust the logged post-apply text. ``apply_diffs`` is
            # deterministic, but logging the text directly keeps replay
            # robust to applier-policy changes.
            state.world_model = state.world_model.with_text(
                event.new_world_model_text
            )

        case ExplanationFailed():
            if state.in_flight_explanation_request_id == event.request_id:
                state.in_flight_explanation_request_id = None

        case OuterStepCompleted():
            if state.hypothesis is not None:
                state.completed_hypotheses.append(state.hypothesis)
            state.hypothesis = None
            state.result = None
            state.explanation = None
            state.in_flight_hypothesis_request_id = None
            state.in_flight_experiment_request_id = None
            state.in_flight_explanation_request_id = None
            state.current_step = event.step + 1

    return state


def replay(
    events: Sequence[TheoryEvent[ObservationT]],
) -> TheoryState[ObservationT]:
    """Fold a list of events into the final state. Pure."""
    state: TheoryState[ObservationT] = TheoryState()
    for event in events:
        _ = apply_event(state, event)
    return state


__all__ = ["TheoryState", "apply_event", "replay"]
