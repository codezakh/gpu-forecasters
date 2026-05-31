"""Event-log round-trip + durability tests.

Includes serialization round-trips against a non-trivial parameterized
``ObservationT`` (``TriMulObservation``) to catch Pydantic generic-
inference regressions that wouldn't show up with ``NoFeedback``.
"""

from pathlib import Path

from ulid import ULID

from gpu_forecasters.hill_climbing.domain import Evaluation, NoFeedback, Node
from gpu_forecasters.hill_climbing.scoring_providers.trimul import TriMulObservation
from gpu_forecasters.max_reward_puct.v2.event_log import FileEventLog
from gpu_forecasters.max_reward_puct.v2.events import (
    EvaluationCompleted,
    EvaluationRequested,
    MutationCompleted,
    MutationRequested,
    SearchInitialized,
    StepCompleted,
    StepStarted,
    search_event_adapter,
)
from gpu_forecasters.trimul.core import SuccessFeedback, CaseSpeedup


def _eval(reward: float | None) -> Evaluation[NoFeedback]:
    return Evaluation(observation=NoFeedback(), reward=reward)


def _root() -> Node[NoFeedback]:
    return Node[NoFeedback](
        program_code="0000", evaluation=_eval(0.0), ancestors=[], is_seed=True
    )


def test_append_and_read_round_trip(tmp_path: Path):
    log: FileEventLog[NoFeedback] = FileEventLog(
        tmp_path / "log.jsonl", observation_type=NoFeedback
    )
    root = _root()
    c1 = ULID()
    events = [
        SearchInitialized[NoFeedback](root=root),
        StepStarted(step=0, parent_ulids=[root.ulid]),
        MutationRequested(request_id="m0", parent_ulid=root.ulid),
        MutationCompleted(request_id="m0", code="0001"),
        EvaluationRequested(
            request_id="e0", child_ulid=c1, parent_ulid=root.ulid, code="0001"
        ),
        EvaluationCompleted[NoFeedback](request_id="e0", evaluation=_eval(1.0)),
        StepCompleted(step=0),
    ]
    for e in events:
        log.append(e)

    loaded = log.read_all()
    assert len(loaded) == len(events)
    assert [getattr(e, "kind") for e in loaded] == [
        "search_initialized",
        "step_started",
        "mutation_requested",
        "mutation_completed",
        "evaluation_requested",
        "evaluation_completed",
        "step_completed",
    ]
    init = loaded[0]
    assert isinstance(init, SearchInitialized)
    assert init.root.ulid == root.ulid
    completed = loaded[5]
    assert isinstance(completed, EvaluationCompleted)
    assert completed.evaluation.reward == 1.0


def test_read_all_on_missing_file_returns_empty(tmp_path: Path):
    log: FileEventLog[NoFeedback] = FileEventLog(
        tmp_path / "missing.jsonl", observation_type=NoFeedback
    )
    assert log.read_all() == []


def test_truncated_final_line_is_tolerated(tmp_path: Path):
    path = tmp_path / "log.jsonl"
    log: FileEventLog[NoFeedback] = FileEventLog(path, observation_type=NoFeedback)
    log.append(StepCompleted(step=0))
    log.append(StepCompleted(step=1))
    with open(path, "a") as f:
        _ = f.write('{"kind": "step_com')

    loaded = log.read_all()
    assert len(loaded) == 2


def test_corrupt_non_final_line_raises(tmp_path: Path):
    path = tmp_path / "log.jsonl"
    path.write_text(
        '{"kind": "step_completed", "step": 0}\n'
        "not json at all\n"
        '{"kind": "step_completed", "step": 1}\n'
    )
    log: FileEventLog[NoFeedback] = FileEventLog(path, observation_type=NoFeedback)
    try:
        _ = log.read_all()
    except ValueError as exc:
        assert "Corrupt" in str(exc)
    else:
        raise AssertionError("Expected ValueError for corrupt middle line")


# ---------------------------------------------------------------------------
# Serialization with a non-trivial parameterized ObservationT
# ---------------------------------------------------------------------------


def _trimul_success_eval(reward: float) -> Evaluation[TriMulObservation]:
    """Build an ``Evaluation[TriMulObservation]`` with a realistic
    ``SuccessFeedback`` payload — nested BaseModels, list of another
    BaseModel, the whole thing parameterized on a Generic BaseModel."""
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


def test_trimul_observation_round_trip(tmp_path: Path):
    """Guard against Pydantic dropping generic parameterization when
    the observation type is substantive."""
    root = Node[TriMulObservation](
        program_code="# root",
        evaluation=_trimul_success_eval(1.0),
        ancestors=[],
        is_seed=True,
    )
    c1 = ULID()
    events = [
        SearchInitialized[TriMulObservation](root=root),
        StepStarted(step=0, parent_ulids=[root.ulid]),
        MutationRequested(request_id="m0", parent_ulid=root.ulid),
        MutationCompleted(request_id="m0", code="# child"),
        EvaluationRequested(
            request_id="e0", child_ulid=c1, parent_ulid=root.ulid, code="# child"
        ),
        EvaluationCompleted[TriMulObservation](
            request_id="e0", evaluation=_trimul_success_eval(2.5)
        ),
        StepCompleted(step=0),
    ]

    log: FileEventLog[TriMulObservation] = FileEventLog(
        tmp_path / "trimul.jsonl", observation_type=TriMulObservation
    )
    for e in events:
        log.append(e)

    loaded = log.read_all()
    assert len(loaded) == len(events)

    init = loaded[0]
    assert isinstance(init, SearchInitialized)
    assert init.root.evaluation.reward == 1.0
    root_feedback = init.root.evaluation.observation.feedback
    assert isinstance(root_feedback, SuccessFeedback)
    assert root_feedback.aggregated_speedup == 1.0
    assert len(root_feedback.per_case_speedups) == 1
    assert root_feedback.per_case_speedups[0].seqlen == 256

    completed = loaded[5]
    assert isinstance(completed, EvaluationCompleted)
    assert completed.evaluation.reward == 2.5
    child_feedback = completed.evaluation.observation.feedback
    assert isinstance(child_feedback, SuccessFeedback)
    assert child_feedback.aggregated_speedup == 2.5


def test_adapter_roundtrip_preserves_types():
    """Single-event adapter round-trip as an isolated sanity check."""
    adapter = search_event_adapter(TriMulObservation)
    # NB: must parameterize at construction, not via annotation — Pydantic
    # generic BaseModels otherwise default the TypeVar and silently drop
    # observation fields on serialize.
    evt = EvaluationCompleted[TriMulObservation](
        request_id="e0", evaluation=_trimul_success_eval(3.14)
    )
    blob = adapter.dump_json(evt)
    back = adapter.validate_json(blob)
    assert isinstance(back, EvaluationCompleted)
    assert back.request_id == "e0"
    assert back.evaluation.reward == 3.14


def test_log_written_by_driver_preserves_trimul_observation(tmp_path: Path):
    """Regression: drive a tiny search through the real ``SearchDriver``
    against a TriMulObservation-shaped Evaluation provider and assert the
    on-disk log keeps observation data intact (not ``observation: {}``).

    This is the bug the POC surfaced — the driver was constructing
    ``EvaluationCompleted[ObservationT]`` with the TypeVar instead of
    the concrete class. The fix threads ``observation_type`` through
    the driver. Lock it in here so a regression fails the test suite
    before another live API call.
    """
    from concurrent.futures import Future, ThreadPoolExecutor
    from typing import Self

    from gpu_forecasters.max_reward_puct.v2.config import SearchConfig
    from gpu_forecasters.max_reward_puct.v2.providers import (
        AsyncEvaluationProvider,
        AsyncMutationProvider,
    )
    from gpu_forecasters.max_reward_puct.v2.search import SearchDriver

    class _MutationProvider:
        def __init__(self) -> None:
            self._executor: ThreadPoolExecutor | None = None

        def submit(
            self,
            parent_code: str,
            evaluation: Evaluation[TriMulObservation],
        ) -> Future[str]:
            assert self._executor is not None
            return self._executor.submit(lambda: parent_code + "x")

        def __enter__(self) -> Self:
            self._executor = ThreadPoolExecutor(max_workers=2)
            return self

        def __exit__(self, *args: object) -> None:
            assert self._executor is not None
            self._executor.shutdown(wait=True)

    class _EvalProvider:
        def __init__(self) -> None:
            self._executor: ThreadPoolExecutor | None = None
            self._counter = 0

        def submit(
            self, program_code: str
        ) -> Future[Evaluation[TriMulObservation]]:
            assert self._executor is not None
            self._counter += 1
            r = float(self._counter)
            return self._executor.submit(lambda: _trimul_success_eval(r))

        def __enter__(self) -> Self:
            self._executor = ThreadPoolExecutor(max_workers=2)
            return self

        def __exit__(self, *args: object) -> None:
            assert self._executor is not None
            self._executor.shutdown(wait=True)

    log_path = tmp_path / "log.jsonl"
    event_log: FileEventLog[TriMulObservation] = FileEventLog(
        log_path, observation_type=TriMulObservation
    )
    config = SearchConfig(
        total_budget_steps=1,
        batch_size=1,
        samples_per_parent=1,
        k_per_parent=1,
    )
    with _MutationProvider() as mp, _EvalProvider() as ep:
        mp_typed: AsyncMutationProvider[TriMulObservation] = mp  # pyright: ignore[reportAssignmentType]
        ep_typed: AsyncEvaluationProvider[TriMulObservation] = ep  # pyright: ignore[reportAssignmentType]
        driver = SearchDriver[TriMulObservation](
            config,
            mutation_provider=mp_typed,
            evaluation_provider=ep_typed,
            event_log=event_log,
            observation_type=TriMulObservation,
        )
        _ = driver.run(initial_program="seed")

    # The serialized log must NOT contain ``observation: {}`` anywhere.
    text = log_path.read_text()
    assert '"observation":{}' not in text, (
        "observation field was dropped on serialize — "
        "driver/state isn't using the runtime observation_type."
    )
    # And the round-trip must yield events with intact feedback fields.
    loaded = event_log.read_all()
    eval_completes = [e for e in loaded if isinstance(e, EvaluationCompleted)]
    assert eval_completes, "expected at least one EvaluationCompleted"
    for e in eval_completes:
        assert isinstance(e.evaluation.observation, TriMulObservation)
        assert isinstance(e.evaluation.observation.feedback, SuccessFeedback)


def test_unparameterized_construction_drops_observation():
    """Lock in the gotcha: constructing EvaluationCompleted without
    explicit type subscription silently serializes ``observation`` as
    ``{}``. If this ever starts working transparently, great — delete
    this test. Until then, driver code MUST use the subscribed form.
    """
    adapter = search_event_adapter(TriMulObservation)
    evt = EvaluationCompleted(
        request_id="e0", evaluation=_trimul_success_eval(1.0)
    )
    blob = adapter.dump_json(evt).decode("utf-8")
    assert '"observation":{}' in blob
