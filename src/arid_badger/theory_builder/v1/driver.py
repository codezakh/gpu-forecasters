"""Outer-loop driver for the theory builder.

Reads/writes the durable theory event log, calls the builder + worker,
and folds events into ``TheoryState``. Recovery is straight from the
log: replay, finish whatever step was in flight, then resume the loop.

Crash recovery cases (mid-step):

* Hypothesis was requested but never completed → re-call the builder.
* Hypothesis completed; experiment requested but never completed →
  re-call the worker. The worker's own inner-search log handles
  per-trial recovery.
* Hypothesis + result completed; explanation requested but never
  completed → re-call the builder for the explanation.
* All three completed; outer-step fence not yet emitted → just emit
  the fence.

Builder failures (parse retry exhaustion, infrastructure errors)
surface as ``HypothesisFailed`` / ``ExplanationFailed`` and the loop
moves on. We do NOT retry the inner search itself — that decision is
the worker's, not the driver's.
"""

from __future__ import annotations

from typing import Generic

from loguru import logger
from ulid import ULID

from arid_badger.hill_climbing.domain import ObservationT
from arid_badger.theory_builder.v1.builder import BuilderError
from arid_badger.theory_builder.v1.domain import WorldModel
from arid_badger.theory_builder.v1.event_log import TheoryEventLog
from arid_badger.theory_builder.v1.events import (
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
from arid_badger.theory_builder.v1.interfaces import (
    ExperimentWorker,
    WorldModelBuilder,
)
from arid_badger.theory_builder.v1.state import TheoryState, apply_event


class TheoryBuilderDriver(Generic[ObservationT]):
    """Orchestrates theory event log + builder + worker + reducer.

    One driver per run. Not reusable: callers construct a fresh one.
    """

    def __init__(
        self,
        *,
        builder: WorldModelBuilder[ObservationT],
        worker: ExperimentWorker[ObservationT],
        event_log: TheoryEventLog[ObservationT],
        observation_type: type[ObservationT],
        max_outer_steps: int,
    ) -> None:
        """``observation_type`` is required for the same reason as in
        ``max_reward_puct.v2.SearchDriver``: Pydantic generic
        BaseModels need the concrete type at construction time to
        serialize correctly. Otherwise generic event payloads
        serialize as ``observation: {}``.
        """
        if max_outer_steps < 1:
            raise ValueError("max_outer_steps must be >= 1")
        self._builder = builder
        self._worker = worker
        self._event_log = event_log
        self._observation_type = observation_type
        self._max_outer_steps = max_outer_steps

    # --- Event emission -----------------------------------------------

    def _emit(
        self,
        state: TheoryState[ObservationT],
        event: TheoryEvent[ObservationT],
    ) -> None:
        self._event_log.append(event)
        apply_event(state, event)

    # --- Top-level run ------------------------------------------------

    def run(self, *, initial_world_model: WorldModel) -> TheoryState[ObservationT]:
        # 1. Replay existing log.
        existing_events = self._event_log.read_all()
        state: TheoryState[ObservationT] = TheoryState()
        for event in existing_events:
            apply_event(state, event)

        # 2. Bootstrap if log empty.
        if not existing_events:
            self._emit(
                state,
                TheoryBuildingInitialized(world_model=initial_world_model),
            )

        # 3. Main loop.
        for step in range(state.current_step, self._max_outer_steps):
            self._run_step(state, step)

        return state

    def _run_step(
        self,
        state: TheoryState[ObservationT],
        step: int,
    ) -> None:
        """Run (or finish) one outer step.

        Each phase checks whether a previous run already produced its
        terminal event — i.e. whether the corresponding ``state``
        slot is filled — and skips that phase if so. This is what
        makes the loop idempotent against replay."""
        logger.info("Theory-builder outer step {step} starting.", step=step)

        # --- Phase 1: hypothesis ---
        if state.hypothesis is None:
            self._do_hypothesis_phase(state)

        if state.hypothesis is None:
            # The hypothesis phase failed terminally. Fence the step
            # so we don't get stuck retrying.
            logger.warning(
                "Step {step}: hypothesis phase failed; skipping step.",
                step=step,
            )
            self._emit(state, OuterStepCompleted(step=step))
            return

        # --- Phase 2: experiment ---
        if state.result is None:
            self._do_experiment_phase(state)

        if state.result is None:
            logger.warning(
                "Step {step}: experiment phase failed; skipping step.",
                step=step,
            )
            self._emit(state, OuterStepCompleted(step=step))
            return

        # --- Phase 3: explanation ---
        if state.explanation is None:
            self._do_explanation_phase(state)

        # --- Fence ---
        self._emit(state, OuterStepCompleted(step=step))
        logger.info("Theory-builder outer step {step} done.", step=step)

    # --- Phase 1: propose hypothesis ----------------------------------

    def _do_hypothesis_phase(
        self, state: TheoryState[ObservationT]
    ) -> None:
        request_id = state.in_flight_hypothesis_request_id or str(ULID())
        if state.in_flight_hypothesis_request_id is None:
            self._emit(state, HypothesisRequested(request_id=request_id))
        try:
            hypothesis = self._builder.propose_hypothesis(state.world_model)
        except BuilderError as exc:
            logger.warning("Hypothesis proposal failed: {exc}", exc=exc)
            self._emit(
                state,
                HypothesisFailed(request_id=request_id, reason=str(exc)),
            )
            return
        self._emit(
            state,
            HypothesisCompleted(
                request_id=request_id, hypothesis=hypothesis
            ),
        )

    # --- Phase 2: run experiment --------------------------------------

    def _do_experiment_phase(
        self, state: TheoryState[ObservationT]
    ) -> None:
        assert state.hypothesis is not None
        request_id = state.in_flight_experiment_request_id or str(ULID())
        if state.in_flight_experiment_request_id is None:
            self._emit(
                state,
                ExperimentRequested(
                    request_id=request_id, hypothesis=state.hypothesis
                ),
            )
        try:
            result = self._worker.run(state.hypothesis)
        except Exception as exc:
            logger.warning("Experiment failed: {exc}", exc=exc)
            self._emit(
                state,
                ExperimentFailed(request_id=request_id, reason=repr(exc)),
            )
            return
        # Subscribe with the runtime concrete class — see
        # ``max_reward_puct.v2.search.SearchDriver`` for the same
        # pattern. Constructing as ``ExperimentCompleted[ObservationT]``
        # would parameterize on the TypeVar at runtime and drop the
        # observation on serialize.
        self._emit(
            state,
            ExperimentCompleted[self._observation_type](  # type: ignore[name-defined]
                request_id=request_id, result=result
            ),
        )

    # --- Phase 3: propose explanation ---------------------------------

    def _do_explanation_phase(
        self, state: TheoryState[ObservationT]
    ) -> None:
        assert state.hypothesis is not None
        assert state.result is not None
        request_id = (
            state.in_flight_explanation_request_id or str(ULID())
        )
        if state.in_flight_explanation_request_id is None:
            self._emit(
                state,
                ExplanationRequested[self._observation_type](  # type: ignore[name-defined]
                    request_id=request_id,
                    hypothesis=state.hypothesis,
                    result=state.result,
                ),
            )
        try:
            explanation, new_text = self._builder.propose_explanation(
                state.world_model, state.hypothesis, state.result
            )
        except BuilderError as exc:
            logger.warning("Explanation production failed: {exc}", exc=exc)
            self._emit(
                state,
                ExplanationFailed(
                    request_id=request_id, reason=str(exc)
                ),
            )
            return
        self._emit(
            state,
            ExplanationCompleted(
                request_id=request_id,
                explanation=explanation,
                new_world_model_text=new_text,
            ),
        )


__all__ = ["TheoryBuilderDriver"]
