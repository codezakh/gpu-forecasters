import pytest

from gpu_forecasters.trimul.core import (
    CaseSpeedup,
    CompileFailedFeedback,
    IncorrectFeedback,
    InfrastructureFailureFeedback,
    RuntimeErrorFeedback,
    SuccessFeedback,
)
from gpu_forecasters.ttt_discover.v2.domain.outcome import ParseFailureFeedback
from gpu_forecasters.ttt_discover.v2.scalarizers.by_target_us import ScaleByTargetUs


def _success_with_constant_runtime(runtime_ns: float) -> SuccessFeedback:
    return SuccessFeedback(
        aggregated_speedup=1.0,
        aggregation_method="geomean",
        per_case_speedups=[
            CaseSpeedup(
                seqlen=256,
                bs=2,
                dim=128,
                hiddendim=128,
                nomask=True,
                distribution="normal",
                speedup=1.0,
                runtime_ns=runtime_ns,
                ref_runtime_ns=runtime_ns,
            ),
            CaseSpeedup(
                seqlen=512,
                bs=1,
                dim=128,
                hiddendim=128,
                nomask=True,
                distribution="normal",
                speedup=1.0,
                runtime_ns=runtime_ns,
                ref_runtime_ns=runtime_ns,
            ),
        ],
    )


def test_success_at_target_is_one() -> None:
    out = _success_with_constant_runtime(runtime_ns=2_500_000.0)  # 2500us
    assert ScaleByTargetUs(target_us=2500.0).scalarize(out) == pytest.approx(1.0)


def test_faster_than_target_is_above_one() -> None:
    out = _success_with_constant_runtime(runtime_ns=1_250_000.0)  # 1250us
    assert ScaleByTargetUs(target_us=2500.0).scalarize(out) == pytest.approx(2.0)


def test_failures_all_zero() -> None:
    scalarizer = ScaleByTargetUs(target_us=2500.0)
    assert scalarizer.scalarize(None) == 0.0
    assert scalarizer.scalarize(ParseFailureFeedback(reason="x")) == 0.0
    assert scalarizer.scalarize(CompileFailedFeedback(compilation_error="x")) == 0.0
    assert (
        scalarizer.scalarize(
            RuntimeErrorFeedback(
                runtime_error_name="ValueError", runtime_error="x", traceback="x"
            )
        )
        == 0.0
    )
    assert scalarizer.scalarize(IncorrectFeedback(error_message="x")) == 0.0
    assert (
        scalarizer.scalarize(InfrastructureFailureFeedback(reason="modal")) == 0.0
    )


def test_empty_per_case_success_is_zero() -> None:
    out = SuccessFeedback(
        aggregated_speedup=0.0,
        aggregation_method="geomean",
        per_case_speedups=[],
    )
    assert ScaleByTargetUs(target_us=2500.0).scalarize(out) == 0.0


def test_rejects_non_positive_target() -> None:
    with pytest.raises(ValueError):
        _ = ScaleByTargetUs(target_us=0.0)
