"""``ExperimentWorker`` adapter over ``max_reward_puct.v2.SearchDriver``.

One outer-loop step constructs a fresh worker. The worker:

1. Builds a hypothesis-conditioned mutation provider for the
   hypothesis it was given.
2. Constructs a fresh v2 ``SearchDriver`` against a per-step
   ``FileEventLog`` (``inner_events.jsonl`` under ``run_dir/step_NN/``).
3. Runs the inner search to budget exhaustion.
4. Folds every evaluation in the log into an ``ExperimentResult``.

The per-step inner-search log is independent of the outer-loop log.
That way, recovery of an outer step that crashed mid-inner-search
re-uses the inner driver's own crash-recovery story (replay log,
re-dispatch un-terminated requests).

Construction args separate "what's fixed for this run" (the worker
factory) from "what changes per step" (the hypothesis). The factory
is a small object so the driver doesn't have to recreate provider
configs each step.
"""

from __future__ import annotations

from pathlib import Path
from typing import Self

from loguru import logger
from pydantic import BaseModel, ConfigDict

from arid_badger.hill_climbing.scoring_providers.trimul import TriMulObservation
from arid_badger.max_reward_puct.v2.config import SearchConfig
from arid_badger.max_reward_puct.v2.event_log import FileEventLog
from arid_badger.max_reward_puct.v2.events import (
    EvaluationCompleted,
    EvaluationRequested,
)
from arid_badger.max_reward_puct.v2.providers import (
    AsyncEvaluationProvider,
    AsyncMutationProvider,
)
from arid_badger.max_reward_puct.v2.search import SearchDriver
from arid_badger.theory_builder.v1.domain import (
    ExperimentResult,
    ExperimentTrial,
    Hypothesis,
)
from arid_badger.theory_builder.v1.mutation_provider import (
    HypothesisConditionedTriMulMutationProvider,
)


class TriMulWorkerConfig(BaseModel):
    """All knobs the worker needs that don't change per outer step."""

    model_config = ConfigDict(frozen=True)

    inner_search: SearchConfig
    initial_program: str
    model_slug: str
    gpu_name: str
    triton_version: str = "3.3.1"
    max_llm_concurrency: int = 8
    request_timeout_s: float = 300.0
    max_tokens: int | None = None
    temperature: float = 1.0


class TriMulExperimentWorker:
    """Adapter implementing ``ExperimentWorker[TriMulObservation]``.

    The Modal evaluation provider is injected; constructing it per
    outer step is wasteful (each ``__enter__`` opens a Modal session)
    and hides the lifecycle — the driver should own the session and
    pass it in. The mutation provider IS constructed per call to
    ``run`` because it carries the hypothesis as state.
    """

    def __init__(
        self,
        *,
        config: TriMulWorkerConfig,
        evaluation_provider: AsyncEvaluationProvider[TriMulObservation],
        run_dir: Path,
    ) -> None:
        self._config = config
        self._evaluation_provider = evaluation_provider
        self._run_dir = run_dir
        self._step_counter = 0

    def __enter__(self) -> Self:
        self._run_dir.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        return None

    def run(
        self, hypothesis: Hypothesis
    ) -> ExperimentResult[TriMulObservation]:
        # Per-step inner-search log lives next to the outer-loop log.
        # The step counter monotonically increments across the worker's
        # lifetime and is used purely as a directory name; it has no
        # bearing on outer-loop state.
        step_dir = self._run_dir / f"step_{self._step_counter:04d}"
        step_dir.mkdir(parents=True, exist_ok=True)
        self._step_counter += 1

        log_path = step_dir / "inner_events.jsonl"
        event_log: FileEventLog[TriMulObservation] = FileEventLog(
            log_path, observation_type=TriMulObservation
        )

        mutation_provider = HypothesisConditionedTriMulMutationProvider(
            hypothesis=hypothesis,
            model_slug=self._config.model_slug,
            gpu_name=self._config.gpu_name,
            triton_version=self._config.triton_version,
            max_llm_concurrency=self._config.max_llm_concurrency,
            request_timeout_s=self._config.request_timeout_s,
            max_tokens=self._config.max_tokens,
            temperature=self._config.temperature,
        )

        logger.info(
            "Theory-builder inner search starting: hypothesis_id={hid} step_dir={dir}",
            hid=str(hypothesis.id),
            dir=step_dir,
        )

        with mutation_provider:
            mutation_typed: AsyncMutationProvider[TriMulObservation] = (
                mutation_provider  # pyright: ignore[reportAssignmentType]
            )
            driver = SearchDriver[TriMulObservation](
                self._config.inner_search,
                mutation_provider=mutation_typed,
                evaluation_provider=self._evaluation_provider,
                event_log=event_log,
                observation_type=TriMulObservation,
            )
            _ = driver.run(initial_program=self._config.initial_program)

        # Fold the inner-search log into an ExperimentResult by
        # picking out every EvaluationCompleted event. We use the
        # stored event log directly (rather than the in-memory
        # archive) so the trial bag includes failures and intermediate
        # children — those are exactly the data the builder needs to
        # reason about why something didn't work.
        trials: list[ExperimentTrial[TriMulObservation]] = []
        eval_request_to_code: dict[str, str] = {}

        for event in event_log.read_all():
            if isinstance(event, EvaluationRequested):
                eval_request_to_code[event.request_id] = event.code
            elif isinstance(event, EvaluationCompleted):
                code = eval_request_to_code.get(event.request_id)
                if code is None:
                    continue
                trials.append(
                    ExperimentTrial[TriMulObservation](
                        code=code, evaluation=event.evaluation
                    )
                )

        result = ExperimentResult[TriMulObservation](
            hypothesis_id=hypothesis.id, trials=trials
        )
        logger.info(
            "Theory-builder inner search done: hypothesis_id={hid} "
            "trials={n} valid={v} best={best}",
            hid=str(hypothesis.id),
            n=result.num_trials,
            v=result.num_valid_trials,
            best=(
                f"{result.best_trial.evaluation.reward:.3f}x"
                if result.best_trial is not None
                and result.best_trial.evaluation.reward is not None
                else "none"
            ),
        )
        return result


__all__ = [
    "TriMulWorkerConfig",
    "TriMulExperimentWorker",
]
