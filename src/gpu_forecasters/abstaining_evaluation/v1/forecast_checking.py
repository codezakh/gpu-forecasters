"""Check surrogate forecasts against ground truth from a real evaluator.

The compound search loop short-circuits the GPU evaluation when the
abstaining surrogate forecasts. The search proceeds with the surrogate's
``expected_speedup`` as the reward, and the kernel never reaches the
real evaluator. After the run, we want to know:

* how well the surrogate's forecasts tracked reality on the candidates
  it chose to predict (calibration along the search trajectory);
* whether the run's headline ``best_forecast_reward`` survives ground
  truth (confirmation/refutation of the search's claim).

This module provides the offline tool: walk the event log, find every
``ForecastObservation`` completion, run the same code through a real
evaluator, and emit one :class:`CheckedForecast` per forecast.

Design split (per the architecture's plan/DSL pattern):

* :func:`forecasts_to_check` is a *pure* function over events. It joins
  ``EvaluationRequested.code`` with each ``ForecastObservation``
  completion via ``request_id`` and yields the synthesized
  :class:`Node` s ready to be re-evaluated. Unit-testable in isolation.
* :class:`ForecastChecker` is the *executor* that consumes those nodes
  and routes each one through an injected real evaluator, with a
  per-ulid filesystem cache so the tool is restartable. In-memory
  dedup by code hash collapses redundant GPU calls within a single run.
* :func:`load_checked_forecasts` reads the cache directory after the
  fact — comparison renderers use this without re-running anything.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Sequence
from concurrent.futures import Future, wait
from pathlib import Path
from typing import Generic, cast

from loguru import logger
from pydantic import BaseModel, ConfigDict
from ulid import ULID

from gpu_forecasters.abstaining_evaluation.v1.observation import (
    CompoundObservation,
    ForecastObservation,
)
from gpu_forecasters.gpu_mode_kernel.core import (
    CaseSpeedupT,
    GpuModeKernelObservation,
    InfrastructureFailureFeedback,
)
from gpu_forecasters.hill_climbing.domain import Evaluation, Node
from gpu_forecasters.landscape_map.v2 import KernelRuntimeEstimate
from gpu_forecasters.max_reward_puct.v2.events import (
    EvaluationCompleted,
    EvaluationRequested,
    SearchEvent,
)
from gpu_forecasters.max_reward_puct.v2.providers import AsyncEvaluationProvider


class CheckedForecast(BaseModel, Generic[CaseSpeedupT]):
    """A surrogate forecast paired with the real evaluator's verdict.

    Constructed once per forecasted candidate after the run, and
    written to ``{cache_dir}/{child_ulid}.json``. ``real_reward`` is
    ``None`` when the real evaluator failed (infrastructure error, or
    a candidate that the GPU couldn't time — e.g. compile failure);
    in that case ``real_observation.feedback`` carries the failure arm
    and downstream code should treat the row as a refutation rather
    than a usable calibration point.
    """

    model_config = ConfigDict(frozen=True)

    child_ulid: ULID
    code_sha256: str
    forecast_reward: float
    forecast_estimate: KernelRuntimeEstimate
    real_observation: GpuModeKernelObservation[CaseSpeedupT]
    real_reward: float | None


# ---------------------------------------------------------------------------
# Pure planning step
# ---------------------------------------------------------------------------


def forecasts_to_check(
    events: Sequence[SearchEvent[CompoundObservation[CaseSpeedupT]]],
) -> list[Node[CompoundObservation[CaseSpeedupT]]]:
    """Synthesize one :class:`Node` per ForecastObservation completion.

    The candidate's code is recovered from the paired
    ``EvaluationRequested.code`` via ``request_id``. Returns nodes in
    log order. Pure function — no I/O, fully testable against an
    in-memory event sequence.

    Real-eval completions, failed evaluations, mutations, and search
    bookkeeping events are ignored. ``ancestors`` is populated with
    the immediate parent ulid only; downstream consumers (replay /
    calibration) do not traverse the chain, so we keep this honest
    (one known parent) rather than walking the archive to fabricate
    full ancestry.
    """
    code_by_request: dict[str, EvaluationRequested] = {}
    nodes: list[Node[CompoundObservation[CaseSpeedupT]]] = []

    for event in events:
        if isinstance(event, EvaluationRequested):
            code_by_request[event.request_id] = event
            continue
        if not isinstance(event, EvaluationCompleted):
            continue
        observation = event.evaluation.observation
        if not isinstance(observation, ForecastObservation):
            continue
        requested = code_by_request.get(event.request_id)
        if requested is None:
            # An EvaluationCompleted without a matching EvaluationRequested
            # in the same log violates the v2 driver invariants; surface
            # it loudly rather than silently dropping the forecast.
            raise ValueError(
                f"EvaluationCompleted request_id={event.request_id!r} has no preceding EvaluationRequested in the log"
            )
        node = Node[CompoundObservation[CaseSpeedupT]](
            program_code=requested.code,
            ancestors=[requested.parent_ulid],
            evaluation=event.evaluation,
            ulid=requested.child_ulid,
            is_seed=False,
        )
        nodes.append(node)

    return nodes


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


def _sha256_of(code: str) -> str:
    return hashlib.sha256(code.encode("utf-8")).hexdigest()


def _atomic_write(path: Path, body: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    _ = tmp.write_text(body)
    os.replace(tmp, path)


class ForecastChecker(Generic[CaseSpeedupT]):
    """Re-evaluates forecasted candidates through a real evaluator.

    Construction-time dependencies:

    * ``real_evaluator`` — any ``AsyncEvaluationProvider`` that yields
      ``GpuModeKernelObservation[CaseSpeedupT]``. In production this
      is the same Modal-backed provider the compound run wrapped; in
      tests it can be a stub that resolves futures synchronously.
    * ``cache_dir`` — one JSON file per ``child_ulid``. Atomic writes
      via temp-then-replace; partial files cannot survive a crash.
    * ``case_speedup_type`` — the concrete pack-specific
      ``CaseSpeedup`` subclass. Required to subscript the generic
      ``CheckedForecast`` for serialization, mirroring the v2 driver's
      ``observation_type`` requirement.

    Lifecycle: the real evaluator owns its own context manager. The
    checker does not enter/exit it — callers wrap the real evaluator
    in a ``with`` block before calling :meth:`check`.
    """

    def __init__(
        self,
        *,
        real_evaluator: AsyncEvaluationProvider[GpuModeKernelObservation[CaseSpeedupT]],
        cache_dir: Path,
        case_speedup_type: type[CaseSpeedupT],
    ) -> None:
        self._real_evaluator = real_evaluator
        self._cache_dir = cache_dir
        self._case_speedup_type = case_speedup_type

    def check(
        self,
        nodes: Sequence[Node[CompoundObservation[CaseSpeedupT]]],
    ) -> list[CheckedForecast[CaseSpeedupT]]:
        """Return one ``CheckedForecast`` per node in input order.

        Already-cached entries are loaded from disk without hitting
        the real evaluator. Among nodes that need a fresh real-eval,
        identical-code candidates share a single submission so the
        GPU is asked at most once per distinct kernel within a call.
        """
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        row_cls = CheckedForecast[self._case_speedup_type]

        results: list[CheckedForecast[CaseSpeedupT] | None] = [None] * len(nodes)
        pending: list[tuple[int, Node[CompoundObservation[CaseSpeedupT]], str]] = []

        # First pass: load cache hits, gather pending work.
        for idx, node in enumerate(nodes):
            cache_path = self._cache_dir / f"{node.ulid}.json"
            if cache_path.exists():
                results[idx] = row_cls.model_validate_json(cache_path.read_text())
                continue
            pending.append((idx, node, _sha256_of(node.program_code)))

        if not pending:
            logger.info(
                "ForecastChecker: all {n} forecast(s) cached at {dir}",
                n=len(nodes),
                dir=self._cache_dir,
            )
            return [cast(CheckedForecast[CaseSpeedupT], r) for r in results]

        # In-memory dedup: at most one submit() per distinct code.
        future_by_hash: dict[str, Future[Evaluation[GpuModeKernelObservation[CaseSpeedupT]]]] = {}
        for _idx, node, code_hash in pending:
            if code_hash in future_by_hash:
                continue
            future_by_hash[code_hash] = self._real_evaluator.submit(node.program_code)

        logger.info(
            "ForecastChecker: {n} forecast(s) -> {u} unique kernel(s); {c} cached, dispatching {d}",
            n=len(nodes),
            u=len(future_by_hash),
            c=len(nodes) - len(pending),
            d=len(future_by_hash),
        )

        # Block on every dispatched future. Per-future failures are
        # captured in the row, not raised — we want a partial cache to
        # survive transient infrastructure errors.
        _ = wait(list(future_by_hash.values()))

        for idx, node, code_hash in pending:
            future = future_by_hash[code_hash]
            real_observation, real_reward = self._materialize(future)
            forecast_obs = node.evaluation.observation
            assert isinstance(forecast_obs, ForecastObservation), (
                f"forecasts_to_check should only emit ForecastObservation"
                f" nodes; got {type(forecast_obs).__name__}"
            )
            row = row_cls(
                child_ulid=node.ulid,
                code_sha256=code_hash,
                forecast_reward=forecast_obs.expected_speedup,
                forecast_estimate=forecast_obs.estimate,
                real_observation=real_observation,
                real_reward=real_reward,
            )
            cache_path = self._cache_dir / f"{node.ulid}.json"
            _atomic_write(cache_path, row.model_dump_json(indent=2))
            results[idx] = row

        return [cast(CheckedForecast[CaseSpeedupT], r) for r in results]

    def _materialize(
        self,
        future: Future[Evaluation[GpuModeKernelObservation[CaseSpeedupT]]],
    ) -> tuple[GpuModeKernelObservation[CaseSpeedupT], float | None]:
        """Resolve one future into (observation, reward).

        On infrastructure failure, fabricate an observation carrying
        ``InfrastructureFailureFeedback`` so the row records what went
        wrong without forcing the caller to handle exceptions.
        """
        try:
            evaluation = future.result()
        except BaseException as exc:
            logger.warning(
                "ForecastChecker: real eval raised; recording as infrastructure failure ({exc!r})",
                exc=exc,
            )
            obs_cls = GpuModeKernelObservation[self._case_speedup_type]
            return (
                obs_cls(
                    feedback=InfrastructureFailureFeedback(reason=repr(exc)),
                    per_case_results=[],
                ),
                None,
            )
        return evaluation.observation, evaluation.reward


# ---------------------------------------------------------------------------
# Result loader
# ---------------------------------------------------------------------------


def load_checked_forecasts(
    cache_dir: Path,
    *,
    case_speedup_type: type[CaseSpeedupT],
) -> list[CheckedForecast[CaseSpeedupT]]:
    """Read every cached row from ``cache_dir``.

    Returns ``[]`` if the directory does not yet exist (a first run
    that hasn't started checking, or a wrong path). Order is
    ulid-sorted, which corresponds to log order since ulids are
    monotonic on creation in the v2 driver.
    """
    if not cache_dir.exists():
        return []
    row_cls = CheckedForecast[case_speedup_type]
    return [
        row_cls.model_validate_json(p.read_text())
        for p in sorted(cache_dir.glob("*.json"))
    ]


__all__ = [
    "CheckedForecast",
    "ForecastChecker",
    "forecasts_to_check",
    "load_checked_forecasts",
]
