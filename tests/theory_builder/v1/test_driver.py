"""Driver integration tests over fakes.

Uses an in-memory event log + a hand-rolled fake builder & worker so
the loop runs in milliseconds. Targets:

* Happy path: a 2-step run records the expected event sequence and
  ends with completed hypotheses + an updated world model.
* Builder failure terminates the step but doesn't abort the run.
* Mid-step crash + restart resumes from the log without redoing
  already-completed phases.
"""

from __future__ import annotations

from typing import Self

from gpu_forecasters.hill_climbing.domain import NoFeedback
from gpu_forecasters.theory_builder.v1.driver import TheoryBuilderDriver
from gpu_forecasters.theory_builder.v1.builder import BuilderError
from gpu_forecasters.theory_builder.v1.domain import (
    Explanation,
    ExperimentResult,
    Hypothesis,
    WorldModel,
)
from gpu_forecasters.theory_builder.v1.event_log import (
    InMemoryTheoryEventLog,
)
from gpu_forecasters.theory_builder.v1.events import (
    ExperimentCompleted,
    ExperimentRequested,
    HypothesisCompleted,
    HypothesisRequested,
    TheoryBuildingInitialized,
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _FakeBuilder:
    """Scripted ``WorldModelBuilder``.

    Constructed with two parallel queues — one for ``propose_hypothesis``
    outputs, one for ``propose_explanation`` outputs. Each queue may
    contain a ``BuilderError`` to simulate retry-exhaustion failure.
    """

    def __init__(
        self,
        *,
        hypothesis_outputs: list[Hypothesis | BuilderError],
        explanation_outputs: list[
            tuple[Explanation, str] | BuilderError
        ],
    ) -> None:
        self._h_outputs = list(hypothesis_outputs)
        self._e_outputs = list(explanation_outputs)
        self.hypothesis_calls = 0
        self.explanation_calls = 0

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def propose_hypothesis(
        self, world_model: WorldModel
    ) -> Hypothesis:
        self.hypothesis_calls += 1
        out = self._h_outputs.pop(0)
        if isinstance(out, BuilderError):
            raise out
        return out

    def propose_explanation(
        self,
        world_model: WorldModel,
        hypothesis: Hypothesis,
        result: ExperimentResult[NoFeedback],
    ) -> tuple[Explanation, str]:
        self.explanation_calls += 1
        out = self._e_outputs.pop(0)
        if isinstance(out, BuilderError):
            raise out
        return out


class _FakeWorker:
    """Scripted ``ExperimentWorker`` — returns one ``ExperimentResult``
    per ``run`` call."""

    def __init__(self, results: list[ExperimentResult[NoFeedback]]) -> None:
        self._results = list(results)
        self.run_calls = 0

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def run(
        self, hypothesis: Hypothesis
    ) -> ExperimentResult[NoFeedback]:
        self.run_calls += 1
        return self._results.pop(0)


def _h(label: str) -> Hypothesis:
    return Hypothesis(
        bottleneck=f"b-{label}",
        intervention=f"i-{label}",
        prediction=f"p-{label}",
        code_references=[],
    )


def _empty_result(h: Hypothesis) -> ExperimentResult[NoFeedback]:
    return ExperimentResult[NoFeedback](
        hypothesis_id=h.id, trials=[]
    )


def _expl(h: Hypothesis) -> Explanation:
    return Explanation(
        hypothesis_id=h.id,
        gap="no gap",
        mechanism="confirmed",
        belief_update="ok",
        diffs=[],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_happy_path_two_steps():
    h1 = _h("1")
    h2 = _h("2")
    builder = _FakeBuilder(
        hypothesis_outputs=[h1, h2],
        explanation_outputs=[
            (_expl(h1), "after-1"),
            (_expl(h2), "after-2"),
        ],
    )
    worker = _FakeWorker(
        [_empty_result(h1), _empty_result(h2)]
    )
    log: InMemoryTheoryEventLog[NoFeedback] = InMemoryTheoryEventLog()

    driver = TheoryBuilderDriver[NoFeedback](
        builder=builder,  # pyright: ignore[reportArgumentType]
        worker=worker,  # pyright: ignore[reportArgumentType]
        event_log=log,
        observation_type=NoFeedback,
        max_outer_steps=2,
    )
    state = driver.run(
        initial_world_model=WorldModel(kernel_description="trimul")
    )

    assert builder.hypothesis_calls == 2
    assert worker.run_calls == 2
    assert builder.explanation_calls == 2

    assert state.current_step == 2
    assert state.world_model.text == "after-2"
    assert [h.bottleneck for h in state.completed_hypotheses] == [
        "b-1",
        "b-2",
    ]

    kinds = [getattr(e, "kind") for e in log.read_all()]
    step_cycle = [
        "hypothesis_requested",
        "hypothesis_completed",
        "experiment_requested",
        "experiment_completed",
        "explanation_requested",
        "explanation_completed",
        "outer_step_completed",
    ]
    expected = ["theory_building_initialized"] + step_cycle * 2
    assert kinds == expected


def test_hypothesis_failure_skips_step_but_continues():
    h2 = _h("2")
    builder = _FakeBuilder(
        hypothesis_outputs=[BuilderError("bad parse"), h2],
        explanation_outputs=[(_expl(h2), "after-2")],
    )
    worker = _FakeWorker([_empty_result(h2)])
    log: InMemoryTheoryEventLog[NoFeedback] = InMemoryTheoryEventLog()

    driver = TheoryBuilderDriver[NoFeedback](
        builder=builder,  # pyright: ignore[reportArgumentType]
        worker=worker,  # pyright: ignore[reportArgumentType]
        event_log=log,
        observation_type=NoFeedback,
        max_outer_steps=2,
    )
    state = driver.run(
        initial_world_model=WorldModel(kernel_description="trimul")
    )
    assert state.current_step == 2
    # First step's hypothesis failed, so the worker was only called
    # in step 2.
    assert worker.run_calls == 1
    assert builder.explanation_calls == 1
    # The second step's hypothesis is the only completed one.
    assert state.completed_hypotheses == [h2]

    kinds = [getattr(e, "kind") for e in log.read_all()]
    assert "hypothesis_failed" in kinds
    # Both outer-step fences fired, even for the failed step.
    assert kinds.count("outer_step_completed") == 2


def test_replay_resumes_after_hypothesis_completed_crash():
    """Crash after HypothesisCompleted but before ExperimentRequested.
    A fresh driver should pick up where we left off without redoing
    the hypothesis phase."""
    h = _h("1")
    log: InMemoryTheoryEventLog[NoFeedback] = InMemoryTheoryEventLog()
    log.append(
        TheoryBuildingInitialized(
            world_model=WorldModel(kernel_description="trimul")
        )
    )
    log.append(HypothesisRequested(request_id="r0"))
    log.append(HypothesisCompleted(request_id="r0", hypothesis=h))
    # Crash here.

    builder = _FakeBuilder(
        hypothesis_outputs=[],  # SHOULD NOT be called
        explanation_outputs=[(_expl(h), "after")],
    )
    worker = _FakeWorker([_empty_result(h)])

    driver = TheoryBuilderDriver[NoFeedback](
        builder=builder,  # pyright: ignore[reportArgumentType]
        worker=worker,  # pyright: ignore[reportArgumentType]
        event_log=log,
        observation_type=NoFeedback,
        max_outer_steps=1,
    )
    state = driver.run(
        initial_world_model=WorldModel(kernel_description="trimul")
    )

    assert builder.hypothesis_calls == 0  # we resumed, didn't re-propose
    assert worker.run_calls == 1
    assert builder.explanation_calls == 1
    assert state.world_model.text == "after"
    assert state.current_step == 1


def test_replay_resumes_after_experiment_completed_crash():
    """Crash after ExperimentCompleted but before ExplanationRequested.
    The hypothesis and worker phases should be skipped on resume."""
    h = _h("1")
    result = _empty_result(h)

    log: InMemoryTheoryEventLog[NoFeedback] = InMemoryTheoryEventLog()
    log.append(
        TheoryBuildingInitialized(
            world_model=WorldModel(kernel_description="trimul")
        )
    )
    log.append(HypothesisRequested(request_id="r0"))
    log.append(HypothesisCompleted(request_id="r0", hypothesis=h))
    log.append(ExperimentRequested(request_id="e0", hypothesis=h))
    log.append(
        ExperimentCompleted[NoFeedback](
            request_id="e0", result=result
        )
    )
    # Crash here.

    builder = _FakeBuilder(
        hypothesis_outputs=[],
        explanation_outputs=[(_expl(h), "after")],
    )
    worker = _FakeWorker([])

    driver = TheoryBuilderDriver[NoFeedback](
        builder=builder,  # pyright: ignore[reportArgumentType]
        worker=worker,  # pyright: ignore[reportArgumentType]
        event_log=log,
        observation_type=NoFeedback,
        max_outer_steps=1,
    )
    state = driver.run(
        initial_world_model=WorldModel(kernel_description="trimul")
    )

    assert builder.hypothesis_calls == 0
    assert worker.run_calls == 0
    assert builder.explanation_calls == 1
    assert state.current_step == 1
    assert state.world_model.text == "after"


def test_initialized_event_only_emitted_once():
    """Restart against a non-empty log must not re-emit
    TheoryBuildingInitialized (the world-model would clobber)."""
    h = _h("1")

    log: InMemoryTheoryEventLog[NoFeedback] = InMemoryTheoryEventLog()
    log.append(
        TheoryBuildingInitialized(
            world_model=WorldModel(kernel_description="trimul", text="seed")
        )
    )

    builder = _FakeBuilder(
        hypothesis_outputs=[h],
        explanation_outputs=[(_expl(h), "after")],
    )
    worker = _FakeWorker([_empty_result(h)])
    driver = TheoryBuilderDriver[NoFeedback](
        builder=builder,  # pyright: ignore[reportArgumentType]
        worker=worker,  # pyright: ignore[reportArgumentType]
        event_log=log,
        observation_type=NoFeedback,
        max_outer_steps=1,
    )
    _ = driver.run(
        initial_world_model=WorldModel(
            kernel_description="DIFFERENT", text="DIFFERENT"
        )
    )
    inits = [
        e
        for e in log.read_all()
        if isinstance(e, TheoryBuildingInitialized)
    ]
    assert len(inits) == 1
    # The world model from the *original* init persists; the resumed
    # driver did not overwrite it.
    assert inits[0].world_model.kernel_description == "trimul"
