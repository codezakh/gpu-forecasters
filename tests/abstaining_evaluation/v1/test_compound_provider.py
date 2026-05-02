"""End-to-end test for ``CompoundEvaluationProvider``.

Drives the provider through both arms (forecast and deferral) using
fake surrogate and real-evaluator implementations. The point is to
pin down the routing and Future-resolution semantics — not to
exercise the LLM or Modal layers.
"""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Self

from arid_badger.abstaining_evaluation.v1.forecast_reward import (
    ExpectedSpeedupReward,
)
from arid_badger.abstaining_evaluation.v1.observation import (
    CompoundObservation,
    ForecastObservation,
    RealObservation,
)
from arid_badger.abstaining_evaluation.v1.provider import (
    CompoundEvaluationProvider,
)
from arid_badger.gpu_mode_kernel.core import (
    GpuModeKernelObservation,
    SuccessFeedback,
)
from arid_badger.gpu_mode_kernel.packs.trimul import TriMulCaseSpeedup
from arid_badger.hill_climbing.domain import Evaluation
from arid_badger.landscape_map.v2 import (
    SUCCESS_BINS,
    HardwareContext,
    KernelImplementation,
    KernelRuntimeEstimate,
    KernelRuntimeQuery,
    KernelTaskInfo,
    LlmCallUsage,
    SpeedupBin,
)
from arid_badger.landscape_map.v2.abstain_estimator import (
    AbstainingLlmSpeedupEstimator,
    Deferral,
    Forecast,
    PredictOrDefer,
)


_OBSERVATION_TYPE = CompoundObservation[TriMulCaseSpeedup]


_HARDWARE = HardwareContext(
    device_name="A100-80GB",
    compute_capability=(8, 0),
    total_global_memory_gb=80.0,
    multiprocessor_count=108,
    max_threads_per_multiprocessor=2048,
    clock_rate_ghz=1.41,
    memory_clock_rate_ghz=1.215,
    memory_bus_width_bits=5120,
)


def _delta_estimate(predicted: SpeedupBin) -> KernelRuntimeEstimate:
    probs = {b: 0.0 for b in SUCCESS_BINS}
    probs[predicted] = 1.0
    return KernelRuntimeEstimate(
        predicted_bin=predicted,
        bin_probabilities=probs,
        reasoning="stub",
        raw_probability_sum=1.0,
    )


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _StubAbstainSurrogate(AbstainingLlmSpeedupEstimator):
    """Returns a fixed ``PredictOrDefer`` per call.

    Subclasses ``AbstainingLlmSpeedupEstimator`` to satisfy the
    constructor signature on the compound provider; ``aestimate`` is
    overridden so the parent ``__init__``'s LiteLLM-bound state is
    never exercised.
    """

    def __init__(self, decision: PredictOrDefer) -> None:
        super().__init__(model_slug="stub")
        self._decision = decision

    async def aestimate(  # type: ignore[override]
        self, query: KernelRuntimeQuery
    ) -> tuple[PredictOrDefer, LlmCallUsage | None]:
        # Touch ``query`` so the type-checker sees the parameter is used.
        _ = query.candidate.code
        return self._decision, None


def _success_eval(
    reward: float,
) -> Evaluation[GpuModeKernelObservation[TriMulCaseSpeedup]]:
    feedback = SuccessFeedback[TriMulCaseSpeedup](
        aggregated_speedup=reward,
        aggregation_method="geomean",
        per_case_speedups=[
            TriMulCaseSpeedup(
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
    inner = GpuModeKernelObservation[TriMulCaseSpeedup](
        feedback=feedback, per_case_results=[]
    )
    return Evaluation[GpuModeKernelObservation[TriMulCaseSpeedup]](
        observation=inner, reward=reward
    )


class _StubRealEvaluator:
    """``AsyncEvaluationProvider``-shaped fake. Records every call and
    resolves the future with a deterministic success eval."""

    def __init__(self, *, delay_s: float = 0.0) -> None:
        self._delay_s = delay_s
        self._executor: ThreadPoolExecutor | None = None
        self.calls: list[str] = []

    def submit(
        self, program_code: str
    ) -> Future[Evaluation[GpuModeKernelObservation[TriMulCaseSpeedup]]]:
        assert self._executor is not None
        self.calls.append(program_code)
        # Reward = length of program_code, so different inputs round-trip
        # distinguishably.
        reward = float(len(program_code))

        def _work() -> Evaluation[GpuModeKernelObservation[TriMulCaseSpeedup]]:
            if self._delay_s > 0:
                time.sleep(self._delay_s)
            return _success_eval(reward)

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


# ---------------------------------------------------------------------------
# Wiring
# ---------------------------------------------------------------------------


_TASK = KernelTaskInfo(op_name="trimul", level_id=0, task_id=0)
_REFERENCE = KernelImplementation(
    kernel_name="trimul_pytorch_reference",
    code="# pretend reference",
    runtime_ms=None,
)


def _make_provider(
    *,
    decision: PredictOrDefer,
    real_evaluator: _StubRealEvaluator,
) -> CompoundEvaluationProvider[TriMulCaseSpeedup]:
    return CompoundEvaluationProvider[TriMulCaseSpeedup](
        surrogate=_StubAbstainSurrogate(decision),
        real_evaluator=real_evaluator,  # pyright: ignore[reportArgumentType]
        forecast_reward=ExpectedSpeedupReward(),
        task=_TASK,
        reference=_REFERENCE,
        hardware=_HARDWARE,
        observation_type=_OBSERVATION_TYPE,
        max_surrogate_concurrency=4,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_forecast_path_resolves_without_calling_real_evaluator() -> None:
    forecast_decision = Forecast(estimate=_delta_estimate(SpeedupBin.HIGH_SPEEDUP))
    real_evaluator = _StubRealEvaluator()
    provider = _make_provider(
        decision=forecast_decision, real_evaluator=real_evaluator
    )
    with provider:
        fut = provider.submit("# candidate code")
        evaluation = fut.result(timeout=10.0)

    assert isinstance(evaluation.observation, ForecastObservation)
    assert evaluation.observation.expected_speedup == evaluation.reward
    assert evaluation.observation.estimate.predicted_bin == SpeedupBin.HIGH_SPEEDUP
    # Bin 7 midpoint is 2 ** 1.25.
    assert abs(evaluation.reward - 2.0 ** 1.25) < 1e-9
    # The real evaluator was never asked.
    assert real_evaluator.calls == []


def test_deferral_path_chains_real_evaluator_result() -> None:
    deferral = Deferral(reason="too uncertain to predict")
    real_evaluator = _StubRealEvaluator()
    provider = _make_provider(decision=deferral, real_evaluator=real_evaluator)
    program = "# candidate code that defers"
    with provider:
        fut = provider.submit(program)
        evaluation = fut.result(timeout=10.0)

    assert isinstance(evaluation.observation, RealObservation)
    assert evaluation.observation.deferral_reason == "too uncertain to predict"
    inner_feedback = evaluation.observation.inner.feedback
    assert isinstance(inner_feedback, SuccessFeedback)
    # Reward propagated from the inner evaluation.
    assert evaluation.reward == float(len(program))
    assert real_evaluator.calls == [program]


def test_concurrent_submits_resolve_independently() -> None:
    """Several submits in flight at once must each resolve correctly,
    proving the asyncio loop and outer-Future pattern do not serialize."""
    real_evaluator = _StubRealEvaluator(delay_s=0.05)
    # Surrogate forecasts a different bin than the real evaluator's
    # reward — lets us tell forecast from deferral results apart even
    # when both routes are exercised.
    provider = _make_provider(
        decision=Deferral(reason="batch test"),
        real_evaluator=real_evaluator,
    )
    n = 8
    with provider:
        futures = [provider.submit(f"code-{i}") for i in range(n)]
        results = [f.result(timeout=10.0) for f in futures]

    assert len(real_evaluator.calls) == n
    for i, evaluation in enumerate(results):
        assert isinstance(evaluation.observation, RealObservation)
        assert evaluation.reward == float(len(f"code-{i}"))


def test_submit_before_enter_raises() -> None:
    provider = _make_provider(
        decision=Forecast(estimate=_delta_estimate(SpeedupBin.MINOR_SPEEDUP)),
        real_evaluator=_StubRealEvaluator(),
    )
    try:
        _ = provider.submit("# code")
    except RuntimeError as exc:
        assert "context manager" in str(exc)
    else:
        raise AssertionError("expected RuntimeError when submitting pre-enter")
