"""RetryingSpeedupEstimator tests."""

from __future__ import annotations

import asyncio

import pytest

from arid_badger.landscape_map.v2.domain import (
    KernelImplementation,
    KernelRuntimeEstimate,
    KernelRuntimeQuery,
    KernelTaskInfo,
    LlmCallUsage,
    SpeedupBin,
)
from arid_badger.landscape_map.v2.parsing import EstimatorParseError
from arid_badger.landscape_map.v2.retrying_estimator import RetryingSpeedupEstimator


def _query() -> KernelRuntimeQuery:
    return KernelRuntimeQuery(
        task=KernelTaskInfo(op_name="vector_add", level_id=1, task_id=1),
        reference=KernelImplementation(
            kernel_name="ref", code="pass", runtime_ms=1.0
        ),
        candidate=KernelImplementation(
            kernel_name="cand", code="pass", runtime_ms=1.0
        ),
    )


def _estimate() -> KernelRuntimeEstimate:
    bins = {b: 1.0 / 8 for b in SpeedupBin if b != SpeedupBin.FAILURE}
    return KernelRuntimeEstimate(
        predicted_bin=SpeedupBin.MINOR_SPEEDUP,
        bin_probabilities=bins,
        reasoning="test",
        raw_probability_sum=1.0,
    )


class _FlakyEstimator:
    """Fails the first ``fail_n`` attempts with EstimatorParseError, then succeeds.

    Optionally raises a non-parse exception to test the "don't retry"
    path — set ``raise_non_parse=True`` and the first call will raise
    ``RuntimeError`` instead of ``EstimatorParseError``.
    """

    def __init__(
        self,
        *,
        fail_n: int = 0,
        raise_non_parse: bool = False,
    ) -> None:
        self.fail_n = fail_n
        self.raise_non_parse = raise_non_parse
        self.call_count = 0

    async def aestimate(
        self, query: KernelRuntimeQuery
    ) -> tuple[KernelRuntimeEstimate, LlmCallUsage | None]:
        del query
        self.call_count += 1
        if self.raise_non_parse and self.call_count == 1:
            raise RuntimeError("non-parse failure")
        if self.call_count <= self.fail_n:
            raise EstimatorParseError(
                f"simulated parse failure on attempt {self.call_count}"
            )
        return _estimate(), None


def test_no_failures_passes_through_in_one_call() -> None:
    inner = _FlakyEstimator(fail_n=0)
    wrapper = RetryingSpeedupEstimator(inner, max_retries=2)
    estimate, _ = asyncio.run(wrapper.aestimate(_query()))
    assert estimate.predicted_bin == SpeedupBin.MINOR_SPEEDUP
    assert inner.call_count == 1


def test_retries_until_success() -> None:
    # Inner fails twice, succeeds on the third attempt; max_retries=2
    # gives exactly enough attempts (1 initial + 2 retries = 3 total).
    inner = _FlakyEstimator(fail_n=2)
    wrapper = RetryingSpeedupEstimator(inner, max_retries=2)
    estimate, _ = asyncio.run(wrapper.aestimate(_query()))
    assert estimate.predicted_bin == SpeedupBin.MINOR_SPEEDUP
    assert inner.call_count == 3


def test_exhausts_retries_and_raises_last_parse_error() -> None:
    # Inner always fails; with max_retries=1 we get 2 attempts total
    # and the second parse error propagates.
    inner = _FlakyEstimator(fail_n=10)
    wrapper = RetryingSpeedupEstimator(inner, max_retries=1)
    with pytest.raises(EstimatorParseError) as exc_info:
        asyncio.run(wrapper.aestimate(_query()))
    assert "attempt 2" in str(exc_info.value)
    assert inner.call_count == 2


def test_default_max_retries_is_one() -> None:
    # Default max_retries=1 means 2 total attempts. fail_n=1 succeeds
    # on the second.
    inner = _FlakyEstimator(fail_n=1)
    wrapper = RetryingSpeedupEstimator(inner)
    estimate, _ = asyncio.run(wrapper.aestimate(_query()))
    assert estimate.predicted_bin == SpeedupBin.MINOR_SPEEDUP
    assert inner.call_count == 2


def test_zero_retries_means_one_attempt() -> None:
    inner = _FlakyEstimator(fail_n=1)
    wrapper = RetryingSpeedupEstimator(inner, max_retries=0)
    with pytest.raises(EstimatorParseError):
        asyncio.run(wrapper.aestimate(_query()))
    assert inner.call_count == 1


def test_non_parse_exception_propagates_immediately() -> None:
    inner = _FlakyEstimator(raise_non_parse=True)
    wrapper = RetryingSpeedupEstimator(inner, max_retries=3)
    with pytest.raises(RuntimeError, match="non-parse failure"):
        asyncio.run(wrapper.aestimate(_query()))
    assert inner.call_count == 1


def test_negative_max_retries_rejected() -> None:
    inner = _FlakyEstimator()
    with pytest.raises(ValueError):
        RetryingSpeedupEstimator(inner, max_retries=-1)
