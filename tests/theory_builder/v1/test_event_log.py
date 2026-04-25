"""Event-log round-trip + durability tests for theory-builder events.

Mirrors ``tests/max_reward_puct/v2/test_event_log.py`` for the
analogous reasons: lock in JSONL serialization across the
discriminated union, and confirm a TriMul-shaped ``ExperimentResult``
round-trips intact (no dropped observation fields)."""

from __future__ import annotations

from pathlib import Path

from arid_badger.hill_climbing.domain import Evaluation, NoFeedback
from arid_badger.hill_climbing.scoring_providers.trimul import TriMulObservation
from arid_badger.theory_builder.v1.domain import (
    Explanation,
    ExperimentResult,
    ExperimentTrial,
    Hypothesis,
    WorldModel,
    WorldModelDiff,
)
from arid_badger.theory_builder.v1.event_log import FileTheoryEventLog
from arid_badger.theory_builder.v1.events import (
    ExperimentCompleted,
    ExperimentRequested,
    ExplanationCompleted,
    ExplanationRequested,
    HypothesisCompleted,
    HypothesisRequested,
    OuterStepCompleted,
    TheoryBuildingInitialized,
    theory_event_adapter,
)
from arid_badger.trimul.core import CaseSpeedup, SuccessFeedback


def _h() -> Hypothesis:
    return Hypothesis(
        bottleneck="b",
        intervention="i",
        prediction="p",
        code_references=["x"],
    )


def _trimul_eval(reward: float) -> Evaluation[TriMulObservation]:
    feedback = SuccessFeedback(
        aggregated_speedup=reward,
        aggregation_method="geomean",
        per_case_speedups=[
            CaseSpeedup(
                seqlen=256,
                bs=2,
                dim=128,
                hiddendim=128,
                nomask=True,
                distribution="normal",
                speedup=reward,
                runtime_ns=1000.0,
                ref_runtime_ns=reward * 1000.0,
            ),
        ],
    )
    return Evaluation[TriMulObservation](
        observation=TriMulObservation(feedback=feedback, per_case_results=[]),
        reward=reward,
    )


def test_round_trip_no_feedback(tmp_path: Path):
    log: FileTheoryEventLog[NoFeedback] = FileTheoryEventLog(
        tmp_path / "log.jsonl", observation_type=NoFeedback
    )
    h = _h()
    result: ExperimentResult[NoFeedback] = ExperimentResult[NoFeedback](
        hypothesis_id=h.id, trials=[]
    )
    explanation = Explanation(
        hypothesis_id=h.id,
        gap="g",
        mechanism="m",
        belief_update="u",
        diffs=[WorldModelDiff(search="x", replace="y")],
    )

    events = [
        TheoryBuildingInitialized(
            world_model=WorldModel(kernel_description="trimul")
        ),
        HypothesisRequested(request_id="r0"),
        HypothesisCompleted(request_id="r0", hypothesis=h),
        ExperimentRequested(request_id="e0", hypothesis=h),
        ExperimentCompleted[NoFeedback](request_id="e0", result=result),
        ExplanationRequested[NoFeedback](
            request_id="x0", hypothesis=h, result=result
        ),
        ExplanationCompleted(
            request_id="x0",
            explanation=explanation,
            new_world_model_text="t",
        ),
        OuterStepCompleted(step=0),
    ]
    for e in events:
        log.append(e)

    loaded = log.read_all()
    kinds = [getattr(e, "kind") for e in loaded]
    assert kinds == [
        "theory_building_initialized",
        "hypothesis_requested",
        "hypothesis_completed",
        "experiment_requested",
        "experiment_completed",
        "explanation_requested",
        "explanation_completed",
        "outer_step_completed",
    ]


def test_round_trip_trimul_observation(tmp_path: Path):
    """Guard against Pydantic dropping generic parameterization on
    serialize when the observation type is substantive."""
    h = _h()
    trial = ExperimentTrial[TriMulObservation](
        code="# kernel", evaluation=_trimul_eval(2.5)
    )
    result: ExperimentResult[TriMulObservation] = ExperimentResult[
        TriMulObservation
    ](hypothesis_id=h.id, trials=[trial])

    log: FileTheoryEventLog[TriMulObservation] = FileTheoryEventLog(
        tmp_path / "trimul.jsonl", observation_type=TriMulObservation
    )
    log.append(
        TheoryBuildingInitialized(
            world_model=WorldModel(kernel_description="trimul")
        )
    )
    log.append(HypothesisRequested(request_id="h0"))
    log.append(HypothesisCompleted(request_id="h0", hypothesis=h))
    log.append(ExperimentRequested(request_id="e0", hypothesis=h))
    log.append(
        ExperimentCompleted[TriMulObservation](
            request_id="e0", result=result
        )
    )

    text = (tmp_path / "trimul.jsonl").read_text()
    assert '"observation":{}' not in text, (
        "observation field was dropped — driver code must subscribe "
        "ExperimentCompleted with the runtime observation type."
    )

    loaded = log.read_all()
    completes = [e for e in loaded if isinstance(e, ExperimentCompleted)]
    assert len(completes) == 1
    eval_observation = completes[0].result.trials[0].evaluation.observation
    assert isinstance(eval_observation, TriMulObservation)
    assert isinstance(eval_observation.feedback, SuccessFeedback)
    assert eval_observation.feedback.aggregated_speedup == 2.5


def test_truncated_final_line_is_tolerated(tmp_path: Path):
    path = tmp_path / "log.jsonl"
    log: FileTheoryEventLog[NoFeedback] = FileTheoryEventLog(
        path, observation_type=NoFeedback
    )
    log.append(OuterStepCompleted(step=0))
    log.append(OuterStepCompleted(step=1))
    with open(path, "a") as f:
        _ = f.write('{"kind": "outer_step_compl')

    loaded = log.read_all()
    assert len(loaded) == 2


def test_unparameterized_construction_drops_observation():
    """Lock in the Pydantic gotcha: constructing
    ``ExperimentCompleted`` without a generic subscription silently
    serializes ``observation`` as ``{}``. Driver code MUST use
    ``ExperimentCompleted[runtime_observation_type](...)``.
    """
    adapter = theory_event_adapter(TriMulObservation)
    result = ExperimentResult[TriMulObservation](
        hypothesis_id=_h().id,
        trials=[
            ExperimentTrial[TriMulObservation](
                code="x", evaluation=_trimul_eval(1.0)
            )
        ],
    )
    evt = ExperimentCompleted(request_id="e0", result=result)
    blob = adapter.dump_json(evt).decode("utf-8")
    assert '"observation":{}' in blob
