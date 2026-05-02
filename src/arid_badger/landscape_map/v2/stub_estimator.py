"""Stub estimators for tests and pipeline smoke checks.

Both implementations satisfy the :class:`SpeedupEstimator` protocol
and produce well-formed (renormalized) :class:`KernelRuntimeEstimate`
values without any LLM calls.
"""

from __future__ import annotations

from arid_badger.typing_utils import implements

from .domain import (
    SUCCESS_BINS,
    KernelRuntimeEstimate,
    KernelRuntimeQuery,
    LlmCallUsage,
    SpeedupBin,
    SpeedupEstimator,
)


class StubEstimator:
    """Always predicts a fixed bin with mass concentrated on it.

    Useful as a baseline and for pipeline testing. The distribution
    places probability ``concentrated_mass`` on the fixed bin and
    distributes ``1 - concentrated_mass`` uniformly across the other
    seven success bins.
    """

    def __init__(
        self,
        fixed_bin: SpeedupBin = SpeedupBin.MINOR_SLOWDOWN,
        *,
        concentrated_mass: float = 0.9,
    ) -> None:
        if fixed_bin == SpeedupBin.FAILURE:
            raise ValueError(
                "StubEstimator cannot fix on FAILURE; pick a success bin (1..8)"
            )
        if not 0.0 < concentrated_mass <= 1.0:
            raise ValueError(
                f"concentrated_mass must be in (0, 1], got {concentrated_mass}"
            )
        self._fixed_bin = fixed_bin
        self._concentrated_mass = concentrated_mass

    def estimate(
        self, query: KernelRuntimeQuery
    ) -> tuple[KernelRuntimeEstimate, LlmCallUsage | None]:
        del query
        spread = (1.0 - self._concentrated_mass) / (len(SUCCESS_BINS) - 1)
        bin_probabilities: dict[SpeedupBin, float] = {
            b: (self._concentrated_mass if b == self._fixed_bin else spread)
            for b in SUCCESS_BINS
        }
        return (
            KernelRuntimeEstimate(
                predicted_bin=self._fixed_bin,
                bin_probabilities=bin_probabilities,
                reasoning=f"Stub estimator: always predicts {self._fixed_bin.label}",
                raw_probability_sum=1.0,
            ),
            None,
        )


implements(SpeedupEstimator)(StubEstimator)
