from __future__ import annotations

from arid_badger.typing_utils import implements

from .domain import (
    KernelRuntimeEstimate,
    KernelRuntimeQuery,
    LikertConfidence,
    LlmCallUsage,
    SpeedupBin,
    SpeedupEstimator,
)


class StubEstimator:
    """Always predicts a fixed bin. Useful as a baseline and for pipeline testing."""

    def __init__(self, fixed_bin: SpeedupBin = SpeedupBin.MINOR_SLOWDOWN) -> None:
        self._fixed_bin = fixed_bin

    def estimate(
        self, query: KernelRuntimeQuery
    ) -> tuple[KernelRuntimeEstimate, LlmCallUsage | None]:
        bin_confidences = {
            b: (
                LikertConfidence.VERY_HIGH
                if b == self._fixed_bin
                else LikertConfidence.VERY_LOW
            )
            for b in SpeedupBin
            if b != SpeedupBin.FAILURE
        }
        estimate = KernelRuntimeEstimate(
            predicted_bin=self._fixed_bin,
            bin_confidences=bin_confidences,
            reasoning=f"Stub estimator: always predicts {self._fixed_bin.label}",
        )
        return estimate, None


implements(SpeedupEstimator)(StubEstimator)
