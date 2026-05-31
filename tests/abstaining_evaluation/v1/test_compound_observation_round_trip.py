"""Event-log serialization round-trip for ``CompoundObservation``.

The v2 search driver and event log are generic over the observation
type; this test pins down that ``CompoundObservation[T]``, a
discriminated union of two structurally different arms, survives a
full serialize → JSONL → deserialize cycle without losing the
discriminator or any payload field on either arm.
"""

from __future__ import annotations

from pathlib import Path

from gpu_forecasters.abstaining_evaluation.v1.observation import (
    CompoundObservation,
    ForecastObservation,
    RealObservation,
)
from gpu_forecasters.gpu_mode_kernel.core import (
    GpuModeKernelObservation,
    SuccessFeedback,
)
from gpu_forecasters.gpu_mode_kernel.packs.trimul import TriMulCaseSpeedup
from gpu_forecasters.hill_climbing.domain import Evaluation, Node
from gpu_forecasters.landscape_map.v2 import (
    SUCCESS_BINS,
    KernelRuntimeEstimate,
    SpeedupBin,
)
from gpu_forecasters.max_reward_puct.v2.event_log import FileEventLog
from gpu_forecasters.max_reward_puct.v2.events import (
    EvaluationCompleted,
    SearchInitialized,
    StepCompleted,
    search_event_adapter,
)


_OBSERVATION_TYPE = CompoundObservation[TriMulCaseSpeedup]


def _uniform_estimate() -> KernelRuntimeEstimate:
    """KernelRuntimeEstimate over a uniform distribution on the eight
    success bins. Picks bin 5 as the argmax to stay well-formed."""
    p = 1.0 / 8.0
    return KernelRuntimeEstimate(
        predicted_bin=SpeedupBin.MINOR_SPEEDUP,
        bin_probabilities={b: p for b in SUCCESS_BINS},
        reasoning="uniform prior — test fixture",
        raw_probability_sum=1.0,
    )


def _forecast_eval(reward: float) -> Evaluation[CompoundObservation[TriMulCaseSpeedup]]:
    estimate = _uniform_estimate()
    obs = ForecastObservation(estimate=estimate, expected_speedup=reward)
    return Evaluation[_OBSERVATION_TYPE](observation=obs, reward=reward)  # type: ignore[valid-type]


def _real_eval(
    reward: float, *, deferral_reason: str | None
) -> Evaluation[CompoundObservation[TriMulCaseSpeedup]]:
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
    obs = RealObservation[TriMulCaseSpeedup](
        inner=inner, deferral_reason=deferral_reason
    )
    return Evaluation[_OBSERVATION_TYPE](observation=obs, reward=reward)  # type: ignore[valid-type]


def test_adapter_round_trip_forecast_arm() -> None:
    adapter = search_event_adapter(_OBSERVATION_TYPE)
    evt = EvaluationCompleted[_OBSERVATION_TYPE](  # type: ignore[valid-type]
        request_id="e0", evaluation=_forecast_eval(1.42)
    )
    blob = adapter.dump_json(evt)
    back = adapter.validate_json(blob)
    assert isinstance(back, EvaluationCompleted)
    obs = back.evaluation.observation
    assert isinstance(obs, ForecastObservation)
    assert obs.kind == "forecast"
    assert obs.expected_speedup == 1.42
    assert obs.estimate.predicted_bin == SpeedupBin.MINOR_SPEEDUP


def test_adapter_round_trip_real_arm() -> None:
    adapter = search_event_adapter(_OBSERVATION_TYPE)
    evt = EvaluationCompleted[_OBSERVATION_TYPE](  # type: ignore[valid-type]
        request_id="e0",
        evaluation=_real_eval(2.5, deferral_reason="too uncertain"),
    )
    blob = adapter.dump_json(evt)
    back = adapter.validate_json(blob)
    assert isinstance(back, EvaluationCompleted)
    obs = back.evaluation.observation
    assert isinstance(obs, RealObservation)
    assert obs.kind == "real"
    assert obs.deferral_reason == "too uncertain"
    feedback = obs.inner.feedback
    assert isinstance(feedback, SuccessFeedback)
    assert feedback.aggregated_speedup == 2.5
    assert len(feedback.per_case_speedups) == 1
    assert feedback.per_case_speedups[0].seqlen == 256


def test_file_event_log_mixed_arms_round_trip(tmp_path: Path) -> None:
    """Both arms in the same log; check structural fields and that
    the on-disk JSONL never contains an empty ``observation`` payload
    (the regression e0114-era code surfaced when the driver wasn't
    threading ``observation_type`` through to event subscription)."""
    log_path = tmp_path / "compound.jsonl"
    log: FileEventLog[CompoundObservation[TriMulCaseSpeedup]] = FileEventLog(
        log_path, observation_type=_OBSERVATION_TYPE
    )

    root = Node[_OBSERVATION_TYPE](  # type: ignore[valid-type]
        program_code="# seed",
        evaluation=_real_eval(1.0, deferral_reason=None),
        ancestors=[],
        is_seed=True,
    )
    forecast_eval_evt = EvaluationCompleted[_OBSERVATION_TYPE](  # type: ignore[valid-type]
        request_id="e_forecast", evaluation=_forecast_eval(1.7)
    )
    real_eval_evt = EvaluationCompleted[_OBSERVATION_TYPE](  # type: ignore[valid-type]
        request_id="e_real",
        evaluation=_real_eval(3.0, deferral_reason="low max-prob"),
    )
    events = [
        SearchInitialized[_OBSERVATION_TYPE](root=root),  # type: ignore[valid-type]
        forecast_eval_evt,
        real_eval_evt,
        StepCompleted(step=0),
    ]
    for e in events:
        log.append(e)

    on_disk = log_path.read_text()
    assert '"observation":{}' not in on_disk, (
        "observation field was dropped on serialize — "
        "compound observation type isn't threading through correctly."
    )
    # Both discriminator values must appear in the on-disk log.
    assert '"kind":"forecast"' in on_disk
    assert '"kind":"real"' in on_disk

    loaded = log.read_all()
    assert len(loaded) == len(events)

    init = loaded[0]
    assert isinstance(init, SearchInitialized)
    root_obs = init.root.evaluation.observation
    assert isinstance(root_obs, RealObservation)
    assert root_obs.deferral_reason is None

    forecast = loaded[1]
    assert isinstance(forecast, EvaluationCompleted)
    assert isinstance(forecast.evaluation.observation, ForecastObservation)
    assert forecast.evaluation.observation.expected_speedup == 1.7

    real = loaded[2]
    assert isinstance(real, EvaluationCompleted)
    assert isinstance(real.evaluation.observation, RealObservation)
    assert real.evaluation.observation.deferral_reason == "low max-prob"
    real_feedback = real.evaluation.observation.inner.feedback
    assert isinstance(real_feedback, SuccessFeedback)
    assert real_feedback.aggregated_speedup == 3.0
