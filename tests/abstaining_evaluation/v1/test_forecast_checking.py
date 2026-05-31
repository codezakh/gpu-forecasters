"""Tests for ``forecast_checking``: pure planning + executor.

The planning function ``forecasts_to_check`` is exercised against
hand-built event sequences. The executor ``ForecastChecker`` is
exercised against a stub real-evaluator so the file-cache, in-memory
dedup, and infrastructure-failure capture are pinned without touching
Modal.
"""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Self

from ulid import ULID

from gpu_forecasters.abstaining_evaluation.v1 import (
    CheckedForecast,
    ForecastChecker,
    ForecastObservation,
    ForecastRewardPolicy,
    RealObservation,
    forecasts_to_check,
    load_checked_forecasts,
)
from gpu_forecasters.abstaining_evaluation.v1.forecast_reward import (
    ExpectedSpeedupReward,
)
from gpu_forecasters.abstaining_evaluation.v1.observation import CompoundObservation
from gpu_forecasters.gpu_mode_kernel.core import (
    GpuModeKernelObservation,
    InfrastructureFailureFeedback,
    SuccessFeedback,
)
from gpu_forecasters.gpu_mode_kernel.packs.trimul import TriMulCaseSpeedup
from gpu_forecasters.hill_climbing.domain import Evaluation, Node
from gpu_forecasters.landscape_map.v2 import (
    SUCCESS_BINS,
    KernelRuntimeEstimate,
    SpeedupBin,
)
from gpu_forecasters.max_reward_puct.v2.events import (
    EvaluationCompleted,
    EvaluationFailed,
    EvaluationRequested,
    SearchEvent,
    SearchInitialized,
    StepCompleted,
    StepStarted,
)


_OBSERVATION_TYPE = CompoundObservation[TriMulCaseSpeedup]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _delta_estimate(predicted: SpeedupBin) -> KernelRuntimeEstimate:
    probs = {b: 0.0 for b in SUCCESS_BINS}
    probs[predicted] = 1.0
    return KernelRuntimeEstimate(
        predicted_bin=predicted,
        bin_probabilities=probs,
        reasoning="fixture",
        raw_probability_sum=1.0,
    )


def _forecast_eval(
    estimate: KernelRuntimeEstimate,
    *,
    reward_policy: ForecastRewardPolicy = ExpectedSpeedupReward(),
) -> Evaluation[CompoundObservation[TriMulCaseSpeedup]]:
    reward = reward_policy(estimate)
    obs = ForecastObservation(estimate=estimate, expected_speedup=reward)
    return Evaluation[_OBSERVATION_TYPE](observation=obs, reward=reward)  # type: ignore[valid-type]


def _real_success_eval(
    speedup: float,
) -> Evaluation[CompoundObservation[TriMulCaseSpeedup]]:
    feedback = SuccessFeedback[TriMulCaseSpeedup](
        aggregated_speedup=speedup,
        aggregation_method="geomean",
        per_case_speedups=[
            TriMulCaseSpeedup(
                seqlen=256,
                bs=2,
                dim=128,
                hiddendim=128,
                nomask=True,
                distribution="normal",
                speedup=speedup,
                runtime_ns=1000.0,
                ref_runtime_ns=speedup * 1000.0,
            ),
        ],
    )
    inner = GpuModeKernelObservation[TriMulCaseSpeedup](
        feedback=feedback, per_case_results=[]
    )
    obs = RealObservation[TriMulCaseSpeedup](inner=inner, deferral_reason=None)
    return Evaluation[_OBSERVATION_TYPE](observation=obs, reward=speedup)  # type: ignore[valid-type]


def _request_eval(
    *,
    request_id: str,
    parent_ulid: ULID,
    child_ulid: ULID,
    code: str,
) -> EvaluationRequested:
    return EvaluationRequested(
        request_id=request_id,
        child_ulid=child_ulid,
        parent_ulid=parent_ulid,
        code=code,
    )


def _completed(
    request_id: str, evaluation: Evaluation[CompoundObservation[TriMulCaseSpeedup]]
) -> EvaluationCompleted[CompoundObservation[TriMulCaseSpeedup]]:
    return EvaluationCompleted[_OBSERVATION_TYPE](  # type: ignore[valid-type]
        request_id=request_id, evaluation=evaluation
    )


# ---------------------------------------------------------------------------
# Pure planning tests
# ---------------------------------------------------------------------------


def test_forecasts_to_check_pairs_code_with_forecast_completion() -> None:
    parent = ULID()
    child = ULID()
    estimate = _delta_estimate(SpeedupBin.HIGH_SPEEDUP)
    events: list[SearchEvent[CompoundObservation[TriMulCaseSpeedup]]] = [
        _request_eval(
            request_id="r1",
            parent_ulid=parent,
            child_ulid=child,
            code="# kernel A",
        ),
        _completed("r1", _forecast_eval(estimate)),
    ]

    nodes = forecasts_to_check(events)

    assert len(nodes) == 1
    node = nodes[0]
    assert node.ulid == child
    assert node.program_code == "# kernel A"
    assert node.ancestors == [parent]
    assert node.is_seed is False
    assert isinstance(node.evaluation.observation, ForecastObservation)
    assert node.evaluation.observation.estimate.predicted_bin == SpeedupBin.HIGH_SPEEDUP


def test_forecasts_to_check_skips_real_completions_and_failures() -> None:
    p = ULID()
    forecast_child = ULID()
    real_child = ULID()
    failed_child = ULID()
    events: list[SearchEvent[CompoundObservation[TriMulCaseSpeedup]]] = [
        StepStarted(step=0, parent_ulids=[p]),
        _request_eval(
            request_id="r_forecast",
            parent_ulid=p,
            child_ulid=forecast_child,
            code="# forecast",
        ),
        _completed(
            "r_forecast", _forecast_eval(_delta_estimate(SpeedupBin.MINOR_SPEEDUP))
        ),
        _request_eval(
            request_id="r_real", parent_ulid=p, child_ulid=real_child, code="# real"
        ),
        _completed("r_real", _real_success_eval(2.0)),
        _request_eval(
            request_id="r_fail",
            parent_ulid=p,
            child_ulid=failed_child,
            code="# failed",
        ),
        EvaluationFailed(request_id="r_fail", reason="timeout"),
        StepCompleted(step=0),
    ]

    nodes = forecasts_to_check(events)

    assert [n.ulid for n in nodes] == [forecast_child]
    assert nodes[0].program_code == "# forecast"


def test_forecasts_to_check_preserves_log_order() -> None:
    p = ULID()
    children = [ULID() for _ in range(3)]
    events: list[SearchEvent[CompoundObservation[TriMulCaseSpeedup]]] = []
    for idx, child in enumerate(children):
        rid = f"r{idx}"
        events.append(
            _request_eval(
                request_id=rid,
                parent_ulid=p,
                child_ulid=child,
                code=f"# kernel {idx}",
            )
        )
        events.append(
            _completed(rid, _forecast_eval(_delta_estimate(SpeedupBin.MINOR_SPEEDUP)))
        )

    nodes = forecasts_to_check(events)
    assert [n.ulid for n in nodes] == children


def test_forecasts_to_check_raises_on_dangling_completion() -> None:
    """An EvaluationCompleted without a matching request id is a v2
    invariant violation — the function must surface it loudly, not
    silently drop."""
    events: list[SearchEvent[CompoundObservation[TriMulCaseSpeedup]]] = [
        _completed(
            "no_such_request",
            _forecast_eval(_delta_estimate(SpeedupBin.MINOR_SPEEDUP)),
        ),
    ]
    raised = False
    try:
        _ = forecasts_to_check(events)
    except ValueError as exc:
        assert "no_such_request" in str(exc)
        raised = True
    assert raised, "expected ValueError on dangling completion"


def test_forecasts_to_check_ignores_seed_real_eval() -> None:
    """The bootstrap eval of the seed program is a real eval emitted
    via SearchInitialized rather than EvaluationRequested/Completed,
    and the SearchInitialized event is irrelevant to forecast replay.
    """
    p = ULID()
    seed_root = Node[_OBSERVATION_TYPE](  # type: ignore[valid-type]
        program_code="# seed",
        evaluation=_real_success_eval(1.0),
        ancestors=[],
        is_seed=True,
        ulid=p,
    )
    forecast_child = ULID()
    events: list[SearchEvent[CompoundObservation[TriMulCaseSpeedup]]] = [
        SearchInitialized[_OBSERVATION_TYPE](root=seed_root),  # type: ignore[valid-type]
        _request_eval(
            request_id="r1",
            parent_ulid=p,
            child_ulid=forecast_child,
            code="# child",
        ),
        _completed("r1", _forecast_eval(_delta_estimate(SpeedupBin.HIGH_SPEEDUP))),
    ]

    nodes = forecasts_to_check(events)
    assert [n.ulid for n in nodes] == [forecast_child]


# ---------------------------------------------------------------------------
# Executor tests with stub real evaluator
# ---------------------------------------------------------------------------


class _StubRealEvaluator:
    """``AsyncEvaluationProvider``-shaped stub for unit tests.

    Records every submitted code; resolves each Future on a thread pool
    using a caller-provided result function.
    """

    def __init__(
        self,
        *,
        result_for: dict[str, Evaluation[GpuModeKernelObservation[TriMulCaseSpeedup]]] | None = None,
        raise_for: dict[str, BaseException] | None = None,
        delay_s: float = 0.0,
    ) -> None:
        self._result_for = result_for or {}
        self._raise_for = raise_for or {}
        self._delay_s = delay_s
        self._executor: ThreadPoolExecutor | None = None
        self.calls: list[str] = []

    def submit(
        self, program_code: str
    ) -> Future[Evaluation[GpuModeKernelObservation[TriMulCaseSpeedup]]]:
        assert self._executor is not None
        self.calls.append(program_code)

        def _work() -> Evaluation[GpuModeKernelObservation[TriMulCaseSpeedup]]:
            if self._delay_s > 0:
                time.sleep(self._delay_s)
            if program_code in self._raise_for:
                raise self._raise_for[program_code]
            if program_code in self._result_for:
                return self._result_for[program_code]
            # Default: success with reward = len(code).
            inner = _real_success_eval(float(len(program_code)))
            real_obs = inner.observation
            assert isinstance(real_obs, RealObservation)
            return Evaluation[GpuModeKernelObservation[TriMulCaseSpeedup]](
                observation=real_obs.inner, reward=inner.reward
            )

        return self._executor.submit(_work)

    def __enter__(self) -> Self:
        self._executor = ThreadPoolExecutor(
            max_workers=4, thread_name_prefix="stub-real-eval"
        )
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True)
            self._executor = None


def _make_forecast_node(
    *, code: str, predicted: SpeedupBin
) -> Node[CompoundObservation[TriMulCaseSpeedup]]:
    estimate = _delta_estimate(predicted)
    evaluation = _forecast_eval(estimate)
    return Node[_OBSERVATION_TYPE](  # type: ignore[valid-type]
        program_code=code,
        ancestors=[],
        evaluation=evaluation,
        is_seed=False,
    )


def test_checker_writes_one_row_per_forecast(tmp_path: Path) -> None:
    nodes = [
        _make_forecast_node(code="# kernel A", predicted=SpeedupBin.MINOR_SPEEDUP),
        _make_forecast_node(code="# kernel B", predicted=SpeedupBin.HIGH_SPEEDUP),
    ]
    cache_dir = tmp_path / "cache"

    real = _StubRealEvaluator()
    with real:
        checker = ForecastChecker(
            real_evaluator=real,  # pyright: ignore[reportArgumentType]
            cache_dir=cache_dir,
            case_speedup_type=TriMulCaseSpeedup,
        )
        rows = checker.check(nodes)

    assert len(rows) == 2
    assert sorted(real.calls) == ["# kernel A", "# kernel B"]
    # Each row's real_reward = len(code), the stub's default.
    by_ulid = {r.child_ulid: r for r in rows}
    a = by_ulid[nodes[0].ulid]
    assert a.real_reward == float(len("# kernel A"))
    assert isinstance(a.real_observation.feedback, SuccessFeedback)
    # Cache files are present and load-equivalent.
    cached = load_checked_forecasts(cache_dir, case_speedup_type=TriMulCaseSpeedup)
    assert len(cached) == 2
    assert {r.child_ulid for r in cached} == {n.ulid for n in nodes}


def test_checker_skips_cached_forecasts(tmp_path: Path) -> None:
    node = _make_forecast_node(
        code="# kernel A", predicted=SpeedupBin.SIGNIFICANT_SPEEDUP
    )
    cache_dir = tmp_path / "cache"

    real = _StubRealEvaluator()
    with real:
        checker = ForecastChecker(
            real_evaluator=real,  # pyright: ignore[reportArgumentType]
            cache_dir=cache_dir,
            case_speedup_type=TriMulCaseSpeedup,
        )
        first = checker.check([node])
        first_calls = list(real.calls)
        # Second pass should be a pure cache hit; no fresh submit.
        second = checker.check([node])

    assert len(first_calls) == 1
    assert real.calls == first_calls, (
        "Second checker.check call hit Modal/stub a second time; cache miss"
    )
    assert first[0].real_reward == second[0].real_reward
    assert first[0].child_ulid == second[0].child_ulid


def test_checker_dedups_identical_code_within_run(tmp_path: Path) -> None:
    """Two distinct ulids with the same code should produce two rows
    but only one real-evaluator submission."""
    code = "# duplicate kernel"
    nodes = [
        _make_forecast_node(code=code, predicted=SpeedupBin.MINOR_SPEEDUP),
        _make_forecast_node(code=code, predicted=SpeedupBin.HIGH_SPEEDUP),
    ]
    cache_dir = tmp_path / "cache"

    real = _StubRealEvaluator()
    with real:
        checker = ForecastChecker(
            real_evaluator=real,  # pyright: ignore[reportArgumentType]
            cache_dir=cache_dir,
            case_speedup_type=TriMulCaseSpeedup,
        )
        rows = checker.check(nodes)

    assert real.calls == [code], (
        f"Expected one submit for duplicate code; got {real.calls!r}"
    )
    assert len(rows) == 2
    # Both rows reflect the same real reward but their forecast rewards
    # come from their (distinct) per-node forecast evaluations.
    assert rows[0].code_sha256 == rows[1].code_sha256
    assert rows[0].real_reward == rows[1].real_reward


def test_checker_records_infrastructure_failure_when_real_eval_raises(
    tmp_path: Path,
) -> None:
    code = "# bad kernel"
    node = _make_forecast_node(code=code, predicted=SpeedupBin.MINOR_SPEEDUP)
    cache_dir = tmp_path / "cache"

    real = _StubRealEvaluator(
        raise_for={code: RuntimeError("boom")},
    )
    with real:
        checker = ForecastChecker(
            real_evaluator=real,  # pyright: ignore[reportArgumentType]
            cache_dir=cache_dir,
            case_speedup_type=TriMulCaseSpeedup,
        )
        rows = checker.check([node])

    assert len(rows) == 1
    row = rows[0]
    assert row.real_reward is None
    assert isinstance(row.real_observation.feedback, InfrastructureFailureFeedback)
    assert "boom" in row.real_observation.feedback.reason


def test_load_checked_forecasts_returns_empty_when_dir_missing(
    tmp_path: Path,
) -> None:
    rows = load_checked_forecasts(
        tmp_path / "nope", case_speedup_type=TriMulCaseSpeedup
    )
    assert rows == []


def test_checked_forecast_round_trips_through_json(tmp_path: Path) -> None:
    """Pin that the generic CheckedForecast serializes its observation
    discriminator + estimate without dropping fields."""
    node = _make_forecast_node(
        code="# rt kernel", predicted=SpeedupBin.SIGNIFICANT_SPEEDUP
    )
    cache_dir = tmp_path / "cache"

    real = _StubRealEvaluator()
    with real:
        checker = ForecastChecker(
            real_evaluator=real,  # pyright: ignore[reportArgumentType]
            cache_dir=cache_dir,
            case_speedup_type=TriMulCaseSpeedup,
        )
        _ = checker.check([node])

    cached_path = cache_dir / f"{node.ulid}.json"
    on_disk = cached_path.read_text()
    assert '"feedback": {}' not in on_disk, (
        "real_observation.feedback dropped on serialize — generic"
        " parameter not threading through"
    )
    assert '"kind": "success"' in on_disk

    cls = CheckedForecast[TriMulCaseSpeedup]
    parsed = cls.model_validate_json(on_disk)
    assert parsed.child_ulid == node.ulid
    assert isinstance(parsed.real_observation.feedback, SuccessFeedback)
