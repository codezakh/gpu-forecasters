"""Pure reducer over ``TheoryEvent`` → ``TheoryState``.

Covers status transitions across the three phases, fold semantics
under replay, and explanation-driven world-model updates."""

from __future__ import annotations

from ulid import ULID

from gpu_forecasters.hill_climbing.domain import NoFeedback
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
    ExplanationRequested,
    HypothesisCompleted,
    HypothesisFailed,
    HypothesisRequested,
    OuterStepCompleted,
    TheoryBuildingInitialized,
)
from gpu_forecasters.theory_builder.v1.state import apply_event, replay


def _wm(text: str = "") -> WorldModel:
    return WorldModel(kernel_description="trimul", text=text)


def _h(bottleneck: str = "x") -> Hypothesis:
    return Hypothesis(
        bottleneck=bottleneck,
        intervention="y",
        prediction="z",
        code_references=[],
    )


def _result(hid: ULID) -> ExperimentResult[NoFeedback]:
    return ExperimentResult[NoFeedback](hypothesis_id=hid, trials=[])


def _explanation(hid: ULID) -> Explanation:
    return Explanation(
        hypothesis_id=hid,
        gap="g",
        mechanism="m",
        belief_update="u",
        diffs=[],
    )


def test_initialized_sets_world_model():
    events = [TheoryBuildingInitialized(world_model=_wm("seed"))]
    state = replay(events)
    assert state.world_model.text == "seed"
    assert state.world_model.kernel_description == "trimul"


def test_request_then_complete_clears_in_flight():
    h = _h()
    events = [
        TheoryBuildingInitialized(world_model=_wm()),
        HypothesisRequested(request_id="r0"),
        HypothesisCompleted(request_id="r0", hypothesis=h),
    ]
    state = replay(events)
    assert state.in_flight_hypothesis_request_id is None
    assert state.hypothesis is not None
    assert state.hypothesis.id == h.id


def test_request_then_fail_clears_in_flight_and_leaves_hypothesis_unset():
    events = [
        TheoryBuildingInitialized(world_model=_wm()),
        HypothesisRequested(request_id="r0"),
        HypothesisFailed(request_id="r0", reason="oops"),
    ]
    state = replay(events)
    assert state.in_flight_hypothesis_request_id is None
    assert state.hypothesis is None


def test_in_flight_persists_when_no_terminal_event():
    """Mid-step crash: replay yields state with the request_id still
    in flight, which is exactly what the recovery path keys off."""
    events = [
        TheoryBuildingInitialized(world_model=_wm()),
        HypothesisRequested(request_id="r0"),
    ]
    state = replay(events)
    assert state.in_flight_hypothesis_request_id == "r0"
    assert state.hypothesis is None


def test_full_step_fold():
    h = _h()
    expl = _explanation(h.id)
    events = [
        TheoryBuildingInitialized(world_model=_wm()),
        HypothesisRequested(request_id="h0"),
        HypothesisCompleted(request_id="h0", hypothesis=h),
        ExperimentRequested(request_id="e0", hypothesis=h),
        ExperimentCompleted[NoFeedback](
            request_id="e0", result=_result(h.id)
        ),
        ExplanationRequested[NoFeedback](
            request_id="x0", hypothesis=h, result=_result(h.id)
        ),
        ExplanationCompleted(
            request_id="x0",
            explanation=expl,
            new_world_model_text="updated",
        ),
        OuterStepCompleted(step=0),
    ]
    state = replay(events)
    assert state.current_step == 1
    assert state.world_model.text == "updated"
    assert state.completed_hypotheses == [h]
    assert state.hypothesis is None
    assert state.result is None
    assert state.explanation is None


def test_world_model_text_updated_on_explanation_completed():
    """Expressly checked separately because the new text is logged on
    the ExplanationCompleted event — the reducer must trust it."""
    h = _h()
    state = replay(
        [
            TheoryBuildingInitialized(world_model=_wm("before")),
            HypothesisRequested(request_id="h0"),
            HypothesisCompleted(request_id="h0", hypothesis=h),
            ExperimentRequested(request_id="e0", hypothesis=h),
            ExperimentCompleted[NoFeedback](
                request_id="e0", result=_result(h.id)
            ),
            ExplanationRequested[NoFeedback](
                request_id="x0", hypothesis=h, result=_result(h.id)
            ),
            ExplanationCompleted(
                request_id="x0",
                explanation=_explanation(h.id),
                new_world_model_text="after",
            ),
        ]
    )
    assert state.world_model.text == "after"


def test_step_completed_appends_completed_hypothesis():
    h1 = _h("first")
    h2 = _h("second")
    events = [
        TheoryBuildingInitialized(world_model=_wm()),
        HypothesisRequested(request_id="h0"),
        HypothesisCompleted(request_id="h0", hypothesis=h1),
        ExperimentRequested(request_id="e0", hypothesis=h1),
        ExperimentCompleted[NoFeedback](
            request_id="e0", result=_result(h1.id)
        ),
        ExplanationRequested[NoFeedback](
            request_id="x0", hypothesis=h1, result=_result(h1.id)
        ),
        ExplanationCompleted(
            request_id="x0",
            explanation=_explanation(h1.id),
            new_world_model_text="t1",
        ),
        OuterStepCompleted(step=0),
        HypothesisRequested(request_id="h1"),
        HypothesisCompleted(request_id="h1", hypothesis=h2),
        ExperimentRequested(request_id="e1", hypothesis=h2),
        ExperimentFailed(request_id="e1", reason="modal died"),
        OuterStepCompleted(step=1),
    ]
    state = replay(events)
    assert [hh.bottleneck for hh in state.completed_hypotheses] == ["first", "second"]
    assert state.current_step == 2


def test_apply_event_returns_state_for_chaining():
    state = replay([])
    out = apply_event(
        state, TheoryBuildingInitialized(world_model=_wm("x"))
    )
    assert out is state
    assert state.world_model.text == "x"
