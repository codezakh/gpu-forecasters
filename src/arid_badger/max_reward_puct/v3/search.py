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

The driver bridges the surrogate's ``async`` interface to the
``concurrent.futures`` world with one background asyncio loop. The
loop runs for the lifetime of the driver's context manager; surrogate
calls become ``Future``s via ``run_coroutine_threadsafe`` and enter
the same in-flight tracker as mutation/evaluation futures.
"""

from __future__ import annotations

import asyncio
import threading
from concurrent.futures import FIRST_COMPLETED, Future, wait
from datetime import UTC, datetime
from typing import Any, Generic, Self

from arid_badger.hill_climbing.domain import (
    Evaluation,
    Node,
    ObservationT,
)
from arid_badger.landscape_map.v2 import (
    AsyncSpeedupEstimator,
    HardwareContext,
    KernelImplementation,
    KernelRuntimeQuery,
    KernelTaskInfo,
)
from arid_badger.max_reward_puct.v3.actions import (
    Dispatch,
    EvaluationDispatch,
    ForecastDispatch,
    MutationDispatch,
    compute_pending_actions,
)
from arid_badger.max_reward_puct.v3.config import SearchConfig
from arid_badger.max_reward_puct.v3.event_log import EventLog
from arid_badger.max_reward_puct.v3.events import (
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
from arid_badger.max_reward_puct.v3.providers import (
    AsyncEvaluationProvider,
    AsyncMutationProvider,
)
from arid_badger.max_reward_puct.v3.state import (
    SearchState,
    apply_event,
)


# Default kernel-task identity used in forecast queries when the
# search is not bound to a real KernelBench task (e.g. the binary-
# string smoke test). Real kernel runs construct a proper
# ``KernelTaskInfo`` and pass the matching ``seed_reference_code``.
_PLACEHOLDER_TASK = KernelTaskInfo(op_name="adhoc", level_id=0, task_id=0)


class SearchDriver(Generic[ObservationT]):
    """Orchestrates event log + providers + surrogate + reducer.

    One driver per search run. Not reusable across runs — callers
    construct a fresh one. Use as a context manager so the background
    asyncio loop the surrogate adapter relies on gets started and
    stopped cleanly.

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
        mutation_provider: AsyncMutationProvider[ObservationT],
        evaluation_provider: AsyncEvaluationProvider[ObservationT],
        surrogate: AsyncSpeedupEstimator,
        seed_reference_code: str,
        hardware: HardwareContext,
        event_log: EventLog[ObservationT],
        observation_type: type[ObservationT],
    ) -> None:
        self.config = config
        self.mutation_provider = mutation_provider
        self.evaluation_provider = evaluation_provider
        self.surrogate = surrogate
        self.seed_reference_code = seed_reference_code
        self.hardware = hardware
        self.event_log = event_log
        self._observation_type = observation_type

        self._surrogate_loop: asyncio.AbstractEventLoop | None = None
        self._surrogate_thread: threading.Thread | None = None

    # --- Lifecycle ------------------------------------------------

    def __enter__(self) -> Self:
        loop = asyncio.new_event_loop()
        thread = threading.Thread(
            target=loop.run_forever,
            name="v3-surrogate-loop",
            daemon=True,
        )
        thread.start()
        self._surrogate_loop = loop
        self._surrogate_thread = thread
        return self

    def __exit__(
        self, exc_type: object, exc_val: object, exc_tb: object
    ) -> None:
        if self._surrogate_loop is not None:
            self._surrogate_loop.call_soon_threadsafe(self._surrogate_loop.stop)
        if self._surrogate_thread is not None:
            self._surrogate_thread.join(timeout=5.0)
        self._surrogate_loop = None
        self._surrogate_thread = None

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
        #    the archive has at least one node anyway.
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
                SearchInitialized[self._observation_type](root=root),  # type: ignore[name-defined]
            )

        # 3. Main loop. Pure decision function + I/O interpreter.
        in_flight: dict[Future[Any], Dispatch[ObservationT]] = {}
        in_flight_ids: set[str] = set()

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
                future = self._fire_dispatch(dispatch)
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

    # --- Dispatch helpers ---------------------------------------------

    def _fire_dispatch(
        self,
        dispatch: Dispatch[ObservationT],
    ) -> Future[Any]:
        match dispatch:
            case MutationDispatch():
                return self.mutation_provider.submit(
                    parent_code=dispatch.parent_code,
                    evaluation=dispatch.parent_evaluation,
                )
            case ForecastDispatch():
                return self._fire_forecast(dispatch)
            case EvaluationDispatch():
                return self.evaluation_provider.submit(dispatch.code)

    def _fire_forecast(self, dispatch: ForecastDispatch) -> Future[Any]:
        loop = self._surrogate_loop
        assert loop is not None, (
            "SearchDriver must be used as a context manager — "
            "background asyncio loop has not been started"
        )
        query = KernelRuntimeQuery(
            task=_PLACEHOLDER_TASK,
            reference=KernelImplementation(
                kernel_name="reference",
                code=self.seed_reference_code,
                runtime_ms=None,
            ),
            candidate=KernelImplementation(
                kernel_name="candidate",
                code=dispatch.code,
                runtime_ms=None,
            ),
            hardware=self.hardware,
        )
        coro = self.surrogate.aestimate(query)
        return asyncio.run_coroutine_threadsafe(coro, loop)

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
                # ``aestimate`` returns ``(estimate, usage)``.
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


__all__ = ["SearchDriver"]
