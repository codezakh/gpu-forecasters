"""Driver: turn an event log + providers into a running max-reward PUCT
search.

Responsibilities of this module, and nothing else:

1. Replay the existing event log into a ``SearchState``.
2. If empty, bootstrap by evaluating the root and emitting
   ``SearchInitialized``.
3. For each remaining step, select parents, dispatch ``k`` mutation
   requests per parent, stream evaluation dispatches as each mutation
   lands, drain, emit ``StepCompleted``.
4. On restart mid-step, re-dispatch un-terminated mutation and
   evaluation requests using the stable ``request_id``s in the log,
   then drain to ``StepCompleted``. No work is dropped unless it
   truly never happened before the crash.

Every state transition goes through ``_emit`` which appends to the log
*then* folds into state. The log is authoritative.

Ordering invariant inside a dispatch helper: log the ``Requested`` event
*before* calling ``provider.submit(...)``. The submit call may block on
a semaphore or start network work right away; we need the log entry to
exist before any side effect that could outlive this process so that
recovery can re-dispatch correctly.

Durability cases:

* Crash after append+fsync of a ``Requested`` but before the provider
  call has actually begun — recovery re-dispatches from the logged
  request.
* Crash while the provider's future is in flight — recovery re-dispatches.
* Crash after a future resolves but before the terminal ``Completed``
  event is appended — recovery re-dispatches; the original provider
  call's result is lost, but a fresh call will produce a valid
  replacement.
* Crash during fsync of any event — the truncated last line is dropped
  on read; state is as if the event never happened.
"""

from __future__ import annotations

import time
from concurrent.futures import FIRST_COMPLETED, Future, wait
from datetime import UTC, datetime
from typing import Any, Generic

from loguru import logger
from ulid import ULID

from gpu_forecasters.hill_climbing.domain import (
    Evaluation,
    Node,
    ObservationT,
)
from gpu_forecasters.max_reward_puct.search import select_batch_of_parents
from gpu_forecasters.max_reward_puct.v2.config import SearchConfig
from gpu_forecasters.max_reward_puct.v2.event_log import EventLog
from gpu_forecasters.max_reward_puct.v2.events import (
    EvaluationCompleted,
    EvaluationFailed,
    EvaluationRequested,
    MutationCompleted,
    MutationFailed,
    MutationRequested,
    SearchEvent,
    SearchInitialized,
    StepCompleted,
    StepStarted,
)
from gpu_forecasters.max_reward_puct.v2.providers import (
    AsyncEvaluationProvider,
    AsyncMutationProvider,
)
from gpu_forecasters.max_reward_puct.v2.state import (
    SearchState,
    apply_event,
)


# In-flight future → (kind, request_id, parent_node) tag. The driver's
# main loop uses this to route completions to the correct handler.
_MUTATION = "mutation"
_EVALUATION = "evaluation"


class SearchDriver(Generic[ObservationT]):
    """Orchestrates event log + providers + reducer.

    One driver per search run. Not reusable across runs: callers
    construct a fresh one.
    """

    def __init__(
        self,
        config: SearchConfig,
        *,
        mutation_provider: AsyncMutationProvider[ObservationT],
        evaluation_provider: AsyncEvaluationProvider[ObservationT],
        event_log: EventLog[ObservationT],
        observation_type: type[ObservationT],
    ) -> None:
        """
        ``observation_type`` is required because Pydantic generic
        BaseModels need the concrete type at construction time to
        serialize correctly. ``EvaluationCompleted[ObservationT](...)``
        inside this class would otherwise be parameterized on the
        TypeVar (not the runtime class), and the observation would
        serialize as ``{}``. We use ``observation_type`` to subscribe
        each generic event with the concrete class at dispatch time.
        """
        self.config = config
        self.mutation_provider = mutation_provider
        self.evaluation_provider = evaluation_provider
        self.event_log = event_log
        self._observation_type = observation_type

    # --- Event emission ------------------------------------------------

    def _emit(
        self,
        state: SearchState[ObservationT],
        event: SearchEvent[ObservationT],
    ) -> None:
        """Append to the log, then fold into state. The two steps are
        inseparable: any code path that updates state without logging
        first breaks the 'log is authoritative' invariant."""
        self.event_log.append(event)
        apply_event(
            state,
            event,
            k_per_parent=self.config.k_per_parent,
            archive_capacity=self.config.archive_capacity,
            observation_type=self._observation_type,
        )

    # --- Top-level run -------------------------------------------------

    def run(self, *, initial_program: str) -> SearchState[ObservationT]:
        # 1. Replay existing log.
        state: SearchState[ObservationT] = SearchState()
        for event in self.event_log.read_all():
            apply_event(
                state,
                event,
                k_per_parent=self.config.k_per_parent,
                archive_capacity=self.config.archive_capacity,
                observation_type=self._observation_type,
            )

        # 2. Bootstrap if empty.
        if not state.archive:
            root_eval = self.evaluation_provider.submit(initial_program).result()
            # NB: ``Node[self._observation_type]`` (concrete class), NOT
            # ``Node[ObservationT]`` (TypeVar). The latter parameterizes
            # on the *TypeVar* at runtime, defaults to NoFeedback, and
            # silently drops the observation on serialize.
            root = Node[self._observation_type](  # type: ignore[name-defined]
                program_code=initial_program,
                evaluation=root_eval,
                is_seed=True,
                ancestors=[],
            )
            self._emit(
                state,
                SearchInitialized[self._observation_type](root=root),  # type: ignore[name-defined]
            )

        # 3. Main loop.
        for step in range(state.current_step, self.config.total_budget_steps):
            if state.step_parent_ulids:
                # Recovery: we crashed mid-step. Re-dispatch in-flight
                # requests, drain, then emit StepCompleted.
                logger.info(
                    "Resuming mid-step {step}: re-dispatching {nm} mutation(s) "
                    "and {ne} evaluation(s) from log.",
                    step=step,
                    nm=len(state.in_flight_mutations),
                    ne=sum(
                        1
                        for p in state.in_flight_children.values()
                        if p.evaluation is None and not p.failed
                    ),
                )
                self._recover_step(state)
            else:
                parents = select_batch_of_parents(
                    archive=state.archive,
                    batch_size=self.config.batch_size,
                    visit_counts=state.visit_counts,
                    best_child_rewards=state.best_child_rewards,
                    global_expansion_count=state.global_expansion_count,
                    seed_ids=state.seed_ids,
                    c_puct=self.config.c_puct,
                )
                if not parents:
                    logger.info(
                        "Step {step}: no parents selectable, stopping early.",
                        step=step,
                    )
                    break
                self._emit(
                    state,
                    StepStarted(step=step, parent_ulids=[p.ulid for p in parents]),
                )
                self._run_step(state, parents)

            self._emit(state, StepCompleted(step=step))

        return state

    # --- Step execution ------------------------------------------------

    def _run_step(
        self,
        state: SearchState[ObservationT],
        parents: list[Node[ObservationT]],
    ) -> None:
        """Dispatch ``k`` mutations per parent; stream evaluations as
        each mutation lands; drain to completion."""
        active: dict[Future[Any], tuple[str, str, Node[ObservationT], float]] = {}

        for parent in parents:
            for _ in range(self.config.samples_per_parent):
                request_id = str(ULID())
                fut = self._dispatch_mutation(state, request_id, parent)
                active[fut] = (_MUTATION, request_id, parent, time.monotonic())

        self._drain(state, active)

    def _recover_step(self, state: SearchState[ObservationT]) -> None:
        """Re-dispatch un-terminated requests after a crash."""
        active: dict[Future[Any], tuple[str, str, Node[ObservationT], float]] = {}
        archive_by_ulid = {n.ulid: n for n in state.archive}

        # In-flight mutations split into two cases by ``code``:
        #   * code is None  → mutation never completed, re-submit it.
        #   * code is not None → mutation completed but the eval was
        #     never dispatched; emit the EvaluationRequested now.
        for mutation_request_id, entry in list(state.in_flight_mutations.items()):
            parent_node = archive_by_ulid.get(entry.parent_ulid)
            if parent_node is None:
                self._emit(
                    state,
                    MutationFailed(
                        request_id=mutation_request_id,
                        reason="parent evicted from archive before recovery",
                        completed_at=datetime.now(UTC),
                    ),
                )
                continue
            if entry.code is None:
                fut = self.mutation_provider.submit(
                    parent_code=parent_node.program_code,
                    evaluation=parent_node.evaluation,
                )
                active[fut] = (
                    _MUTATION,
                    mutation_request_id,
                    parent_node,
                    time.monotonic(),
                )
            else:
                eval_request_id, eval_fut = self._dispatch_evaluation(
                    state,
                    parent_node,
                    entry.code,
                    from_mutation_request_id=mutation_request_id,
                )
                active[eval_fut] = (
                    _EVALUATION,
                    eval_request_id,
                    parent_node,
                    time.monotonic(),
                )

        # Re-dispatch in-flight evaluations (EvaluationRequested logged
        # but no terminal yet).
        for request_id, pending in list(state.in_flight_children.items()):
            if pending.evaluation is not None or pending.failed:
                continue
            parent_node = archive_by_ulid.get(pending.parent_ulid)
            if parent_node is None:
                self._emit(
                    state,
                    EvaluationFailed(
                        request_id=request_id,
                        reason="parent evicted from archive before recovery",
                        completed_at=datetime.now(UTC),
                    ),
                )
                continue
            fut = self.evaluation_provider.submit(pending.code)
            active[fut] = (_EVALUATION, request_id, parent_node, time.monotonic())

        self._drain(state, active)

    def _drain(
        self,
        state: SearchState[ObservationT],
        active: dict[Future[Any], tuple[str, str, Node[ObservationT], float]],
    ) -> None:
        """Completion loop: route each finished future to its handler
        until none remain.

        If ``per_request_timeout_s`` is set, ``wait`` is bounded by the
        earliest remaining per-future deadline. When a future crosses
        its deadline we emit a terminal ``*Failed`` for it (so the log
        is consistent and recovery won't retry it) and attempt to
        cancel the underlying future. A future that cannot be cancelled
        is abandoned; it may still complete eventually, but the search
        has already moved on and its result is ignored.
        """
        timeout = self.config.per_request_timeout_s
        while active:
            if timeout is None:
                wait_timeout: float | None = None
            else:
                now = time.monotonic()
                # Earliest deadline across every in-flight future.
                wait_timeout = max(
                    0.0,
                    min(
                        submit_time + timeout - now
                        for (_k, _r, _p, submit_time) in active.values()
                    ),
                )
            done, _ = wait(
                list(active.keys()),
                return_when=FIRST_COMPLETED,
                timeout=wait_timeout,
            )
            if done:
                for fut in done:
                    kind, request_id, parent, _submit_time = active.pop(fut)
                    if kind == _MUTATION:
                        self._handle_mutation_completion(
                            state, request_id, parent, fut, active
                        )
                    else:
                        self._handle_evaluation_completion(state, request_id, fut)
            else:
                # No future completed within the deadline. Expire every
                # future whose per-request budget has elapsed — typically
                # one, but batched if many share the same start tick.
                self._expire_timed_out(state, active)

    def _expire_timed_out(
        self,
        state: SearchState[ObservationT],
        active: dict[Future[Any], tuple[str, str, Node[ObservationT], float]],
    ) -> None:
        timeout = self.config.per_request_timeout_s
        assert timeout is not None, (
            "_expire_timed_out called with no configured timeout; "
            "this is a driver bug"
        )
        now = time.monotonic()
        expired = [
            fut
            for fut, (_k, _r, _p, submit_time) in active.items()
            if now - submit_time >= timeout
        ]
        assert expired, (
            "drain loop entered the timeout branch but no future was "
            "actually past its deadline; this is a driver bug"
        )
        for fut in expired:
            kind, request_id, _parent, _submit_time = active.pop(fut)
            fut.cancel()
            reason = f"timeout after {timeout:.1f}s"
            now_utc = datetime.now(UTC)
            if kind == _MUTATION:
                self._emit(
                    state,
                    MutationFailed(
                        request_id=request_id, reason=reason, completed_at=now_utc
                    ),
                )
            else:
                self._emit(
                    state,
                    EvaluationFailed(
                        request_id=request_id, reason=reason, completed_at=now_utc
                    ),
                )

    # --- Dispatch helpers ---------------------------------------------

    def _dispatch_mutation(
        self,
        state: SearchState[ObservationT],
        request_id: str,
        parent: Node[ObservationT],
    ) -> Future[str]:
        self._emit(
            state,
            MutationRequested(
                request_id=request_id,
                parent_ulid=parent.ulid,
                started_at=datetime.now(UTC),
            ),
        )
        return self.mutation_provider.submit(
            parent_code=parent.program_code,
            evaluation=parent.evaluation,
        )

    def _dispatch_evaluation(
        self,
        state: SearchState[ObservationT],
        parent: Node[ObservationT],
        code: str,
        *,
        from_mutation_request_id: str | None,
    ) -> tuple[str, Future[Evaluation[ObservationT]]]:
        request_id = str(ULID())
        child_ulid = ULID()
        self._emit(
            state,
            EvaluationRequested(
                request_id=request_id,
                child_ulid=child_ulid,
                parent_ulid=parent.ulid,
                code=code,
                from_mutation_request_id=from_mutation_request_id,
                started_at=datetime.now(UTC),
            ),
        )
        fut = self.evaluation_provider.submit(code)
        return request_id, fut

    # --- Completion handlers ------------------------------------------

    def _handle_mutation_completion(
        self,
        state: SearchState[ObservationT],
        request_id: str,
        parent: Node[ObservationT],
        fut: Future[Any],
        active: dict[Future[Any], tuple[str, str, Node[ObservationT], float]],
    ) -> None:
        try:
            code = fut.result()
        except BaseException as exc:
            self._emit(
                state,
                MutationFailed(
                    request_id=request_id,
                    reason=repr(exc),
                    completed_at=datetime.now(UTC),
                ),
            )
            return
        if not isinstance(code, str):
            # Provider contract violation. Fail loudly rather than
            # log a lie.
            self._emit(
                state,
                MutationFailed(
                    request_id=request_id,
                    reason=(
                        f"mutation provider returned {type(code).__name__}, "
                        "expected str"
                    ),
                    completed_at=datetime.now(UTC),
                ),
            )
            return
        self._emit(
            state,
            MutationCompleted(
                request_id=request_id, code=code, completed_at=datetime.now(UTC)
            ),
        )
        eval_request_id, eval_fut = self._dispatch_evaluation(
            state,
            parent,
            code,
            from_mutation_request_id=request_id,
        )
        active[eval_fut] = (_EVALUATION, eval_request_id, parent, time.monotonic())

    def _handle_evaluation_completion(
        self,
        state: SearchState[ObservationT],
        request_id: str,
        fut: Future[Any],
    ) -> None:
        try:
            evaluation = fut.result()
        except BaseException as exc:
            self._emit(
                state,
                EvaluationFailed(
                    request_id=request_id,
                    reason=repr(exc),
                    completed_at=datetime.now(UTC),
                ),
            )
            return
        self._emit(
            state,
            EvaluationCompleted[self._observation_type](  # type: ignore[name-defined]
                request_id=request_id,
                evaluation=evaluation,
                completed_at=datetime.now(UTC),
            ),
        )


__all__ = ["SearchDriver"]
