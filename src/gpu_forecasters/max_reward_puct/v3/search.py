"""v3 max-reward PUCT driver: turn an event log + providers + surrogate
into a running search.

Three responsibilities, each isolated:

1. ``compute_pending_actions(state, config)`` — pure function in
   ``actions.py``; returns events to emit and dispatches to fire.
2. The loop body in ``run`` — interprets actions: emits events, fires
   provider calls, waits for completions.
3. ``_emit(state, event)`` — appends to the log, folds via
   ``apply_event``, returns new state.

Recovery is not a separate code path: replay the log into state, call
``compute_pending_actions``, get the next moves. Same code, fresh or
resumed.

All three providers (mutation, evaluation, surrogate) speak the same
``concurrent.futures.Future`` shape. The surrogate is adapted from
its async-native v2 form by ``CoroutineSpeedupEstimator``; this
driver owns no asyncio loop and imports no async machinery.
"""

from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, Future, wait
from datetime import UTC, datetime
from typing import Any, Generic

from ulid import ULID

from gpu_forecasters.hill_climbing.domain import (
    Evaluation,
    Node,
    ObservationT,
)
from gpu_forecasters.landscape_map.v2 import (
    HardwareContext,
    KernelImplementation,
    KernelRuntimeQuery,
    KernelTaskInfo,
)
from gpu_forecasters.max_reward_puct.v3.actions import (
    Dispatch,
    EvaluationDispatch,
    ForecastDispatch,
    MutationDispatch,
    compute_pending_actions,
)
from gpu_forecasters.max_reward_puct.v3.config import SearchConfig
from gpu_forecasters.max_reward_puct.v3.event_log import EventLog
from gpu_forecasters.max_reward_puct.v3.events import (
    EvaluationCompleted,
    EvaluationFailed,
    EvaluationRequested,
    ForecastCompleted,
    ForecastFailed,
    ForecastRequested,
    MutationCompleted,
    MutationFailed,
    MutationRequested,
    SearchEvent,
    SearchInitialized,
)
from gpu_forecasters.max_reward_puct.v3.providers import (
    EvaluationProvider,
    MutationProvider,
    SpeedupEstimator,
)
from gpu_forecasters.max_reward_puct.v3.state import (
    SearchState,
    apply_event,
)


class SurrogateContextMismatch(Exception):
    """Raised on resume when constructor surrogate-context args don't
    match what's in the log. Refusing to continue is the right call
    here: a different ``seed_reference_code`` or ``hardware`` would
    feed the surrogate inputs incomparable with what the existing
    forecasts in the log were conditioned on, silently corrupting the
    audit trail."""


class SearchDriver(Generic[ObservationT]):
    """Orchestrates event log + providers + surrogate + reducer.

    One driver per search run. Not reusable across runs — callers
    construct a fresh one. The driver owns no resources of its own,
    so it is not a context manager; callers manage the lifecycles of
    the providers and surrogate they pass in.

    ``observation_type`` is required because Pydantic generic
    ``BaseModel``s need the concrete type at construction time to
    serialize correctly. Without it, ``EvaluationCompleted[
    ObservationT](...)`` inside this class parameterizes on the
    *TypeVar*, defaults to ``NoFeedback``, and silently drops the
    observation on serialize.
    """

    def __init__(
        self,
        config: SearchConfig,
        *,
        mutation_provider: MutationProvider[ObservationT],
        evaluation_provider: EvaluationProvider[ObservationT],
        surrogate: SpeedupEstimator,
        kernel_task: KernelTaskInfo,
        seed_reference_code: str,
        hardware: HardwareContext,
        event_log: EventLog[ObservationT],
        observation_type: type[ObservationT],
    ) -> None:
        self.config = config
        self.mutation_provider = mutation_provider
        self.evaluation_provider = evaluation_provider
        self.surrogate = surrogate
        self.kernel_task = kernel_task
        self.seed_reference_code = seed_reference_code
        self.hardware = hardware
        self.event_log = event_log
        self._observation_type = observation_type

    # --- Event emission ------------------------------------------------

    def _emit(
        self,
        state: SearchState[ObservationT],
        event: SearchEvent[ObservationT],
    ) -> SearchState[ObservationT]:
        """Append to the log, then fold into state. The two steps are
        inseparable: any code path that updates state without logging
        first breaks the 'log is authoritative' invariant."""
        self.event_log.append(event)
        return apply_event(
            state,
            event,
            k_per_parent=self.config.k_per_parent,
            archive_capacity=self.config.archive_capacity,
            observation_type=self._observation_type,
        )

    # --- Top-level run -------------------------------------------------

    def run(self, *, initial_program: str) -> SearchState[ObservationT]:
        # 1. Replay log into state.
        state: SearchState[ObservationT] = SearchState()
        for event in self.event_log.read_all():
            state = apply_event(
                state,
                event,
                k_per_parent=self.config.k_per_parent,
                archive_capacity=self.config.archive_capacity,
                observation_type=self._observation_type,
            )

        # 2. Bootstrap if uninitialized. The root eval is synchronous —
        #    the rest of the algorithm has nothing to dispatch until
        #    the archive has at least one node anyway. On resume,
        #    validate constructor surrogate-context args against the
        #    log to catch operator drift early (different reference
        #    code or hardware → forecasts incomparable with what's
        #    already logged).
        if state.is_uninitialized():
            root_eval = self.evaluation_provider.submit(initial_program).result()
            root: Node[ObservationT] = Node[self._observation_type](  # type: ignore[name-defined,valid-type]
                program_code=initial_program,
                evaluation=root_eval,
                is_seed=True,
                ancestors=[],
            )
            state = self._emit(
                state,
                SearchInitialized[self._observation_type](  # type: ignore[name-defined]
                    root=root,
                    kernel_task=self.kernel_task,
                    seed_reference_code=self.seed_reference_code,
                    hardware=self.hardware,
                ),
            )
        else:
            self._validate_surrogate_context(state)

        # 3. Main loop. Pure decision function + I/O interpreter.
        in_flight: dict[Future[Any], Dispatch[ObservationT]] = {}
        in_flight_ids: set[ULID] = set()

        while True:
            actions = compute_pending_actions(state, self.config)
            if actions.is_empty() and not in_flight:
                return state

            for event in actions.events:
                state = self._emit(state, event)

            for dispatch in actions.dispatches:
                if dispatch.request_id in in_flight_ids:
                    # Already firing this provider call in this
                    # process; don't re-fire, don't re-emit.
                    continue
                if not dispatch.is_redispatch:
                    state = self._emit(state, _make_requested_event(dispatch))
                future = self._fire_dispatch(state, dispatch)
                in_flight[future] = dispatch
                in_flight_ids.add(dispatch.request_id)

            if in_flight:
                done_futures, _pending = wait(
                    list(in_flight.keys()), return_when=FIRST_COMPLETED
                )
                for fut in done_futures:
                    dispatch = in_flight.pop(fut)
                    in_flight_ids.discard(dispatch.request_id)
                    state = self._handle_completion(state, dispatch, fut)

    # --- Resume validation --------------------------------------------

    def _validate_surrogate_context(
        self, state: SearchState[ObservationT]
    ) -> None:
        """Refuse to resume if constructor args don't match the log's
        surrogate context. Replaying produces a state where these are
        non-None (set by the SearchInitialized reducer arm); any
        difference means the surrogate would be conditioned on
        different inputs than the existing forecasts in the log."""
        mismatches: list[str] = []
        if state.kernel_task != self.kernel_task:
            mismatches.append(
                f"kernel_task: log={state.kernel_task!r} "
                f"constructor={self.kernel_task!r}"
            )
        if state.seed_reference_code != self.seed_reference_code:
            mismatches.append(
                f"seed_reference_code: log={state.seed_reference_code!r} "
                f"constructor={self.seed_reference_code!r}"
            )
        if state.hardware != self.hardware:
            mismatches.append(
                f"hardware: log={state.hardware!r} "
                f"constructor={self.hardware!r}"
            )
        if mismatches:
            raise SurrogateContextMismatch(
                "Refusing to resume: surrogate context differs from log.\n"
                + "\n".join(f"  - {m}" for m in mismatches)
            )

    # --- Dispatch helpers ---------------------------------------------

    def _fire_dispatch(
        self,
        state: SearchState[ObservationT],
        dispatch: Dispatch[ObservationT],
    ) -> Future[Any]:
        match dispatch:
            case MutationDispatch():
                return self.mutation_provider.submit(
                    parent_code=dispatch.parent_code,
                    evaluation=dispatch.parent_evaluation,
                )
            case ForecastDispatch():
                # State carries the surrogate-conditioning constants
                # set at SearchInitialized; reading them here keeps
                # the log authoritative even if the driver instance
                # was constructed with stale values.
                assert state.kernel_task is not None
                assert state.seed_reference_code is not None
                assert state.hardware is not None
                query = KernelRuntimeQuery(
                    task=state.kernel_task,
                    reference=KernelImplementation(
                        kernel_name="reference",
                        code=state.seed_reference_code,
                        runtime_ms=None,
                    ),
                    candidate=KernelImplementation(
                        kernel_name="candidate",
                        code=dispatch.code,
                        runtime_ms=None,
                    ),
                    hardware=state.hardware,
                )
                return self.surrogate.submit(query)
            case EvaluationDispatch():
                return self.evaluation_provider.submit(dispatch.code)

    # --- Completion handlers ------------------------------------------

    def _handle_completion(
        self,
        state: SearchState[ObservationT],
        dispatch: Dispatch[ObservationT],
        future: Future[Any],
    ) -> SearchState[ObservationT]:
        now_utc = datetime.now(UTC)
        try:
            result: Any = future.result()
        except BaseException as exc:  # noqa: BLE001 — provider errors must surface as events
            return self._emit(state, _make_failed_event(dispatch, repr(exc), now_utc))

        match dispatch:
            case MutationDispatch():
                if not isinstance(result, str):
                    return self._emit(
                        state,
                        MutationFailed(
                            step=dispatch.step,
                            request_id=dispatch.request_id,
                            parent_ulid=dispatch.parent_ulid,
                            reason=(
                                f"mutation provider returned "
                                f"{type(result).__name__}, expected str"
                            ),
                            completed_at=now_utc,
                        ),
                    )
                return self._emit(
                    state,
                    MutationCompleted(
                        step=dispatch.step,
                        request_id=dispatch.request_id,
                        parent_ulid=dispatch.parent_ulid,
                        code=result,
                        completed_at=now_utc,
                    ),
                )
            case ForecastDispatch():
                # ``submit`` returns ``(estimate, usage)``.
                estimate, usage = result
                return self._emit(
                    state,
                    ForecastCompleted(
                        step=dispatch.step,
                        request_id=dispatch.request_id,
                        parent_ulid=dispatch.parent_ulid,
                        forecast=estimate,
                        llm_usage=usage,
                        completed_at=now_utc,
                    ),
                )
            case EvaluationDispatch():
                evaluation: Evaluation[ObservationT] = result
                return self._emit(
                    state,
                    EvaluationCompleted[self._observation_type](  # type: ignore[name-defined]
                        step=dispatch.step,
                        request_id=dispatch.request_id,
                        parent_ulid=dispatch.parent_ulid,
                        evaluation=evaluation,
                        completed_at=now_utc,
                    ),
                )


def _make_requested_event(
    dispatch: Dispatch[Any],
) -> SearchEvent[Any]:
    """Build the ``Requested`` event paired with a fresh dispatch."""
    started_at = datetime.now(UTC)
    match dispatch:
        case MutationDispatch():
            return MutationRequested(
                step=dispatch.step,
                request_id=dispatch.request_id,
                parent_ulid=dispatch.parent_ulid,
                started_at=started_at,
            )
        case ForecastDispatch():
            return ForecastRequested(
                step=dispatch.step,
                request_id=dispatch.request_id,
                parent_ulid=dispatch.parent_ulid,
                code=dispatch.code,
                started_at=started_at,
            )
        case EvaluationDispatch():
            return EvaluationRequested(
                step=dispatch.step,
                request_id=dispatch.request_id,
                parent_ulid=dispatch.parent_ulid,
                code=dispatch.code,
                started_at=started_at,
            )


def _make_failed_event(
    dispatch: Dispatch[Any],
    reason: str,
    completed_at: datetime,
) -> SearchEvent[Any]:
    """Build the matching ``Failed`` event for a dispatch whose
    underlying future raised."""
    match dispatch:
        case MutationDispatch():
            return MutationFailed(
                step=dispatch.step,
                request_id=dispatch.request_id,
                parent_ulid=dispatch.parent_ulid,
                reason=reason,
                completed_at=completed_at,
            )
        case ForecastDispatch():
            return ForecastFailed(
                step=dispatch.step,
                request_id=dispatch.request_id,
                parent_ulid=dispatch.parent_ulid,
                reason=reason,
                completed_at=completed_at,
            )
        case EvaluationDispatch():
            return EvaluationFailed(
                step=dispatch.step,
                request_id=dispatch.request_id,
                parent_ulid=dispatch.parent_ulid,
                reason=reason,
                completed_at=completed_at,
            )


__all__ = ["SearchDriver", "SurrogateContextMismatch"]
