"""Retry decorator for ``AsyncSpeedupEstimator``.

Wraps any inner async estimator and retries the call when it raises
:class:`EstimatorParseError`. Other exceptions (network errors,
timeouts, validation errors that aren't parse-shaped) propagate
unchanged — they belong at a different layer.

The motivating failure mode is the gpt-oss "answer-in-channel" mode,
where the model emits its answer in the ``final`` channel instead of
calling the tool. With temperature 1.0 sampling this is roughly
i.i.d. across attempts, so one retry knocks a ~10% per-call failure
rate down to ~1%, and two retries to ~0.1%.

Retries are below the v3 driver, so the v3 event log only sees the
final outcome (one ForecastCompleted on success, one ForecastFailed
on terminal failure). The per-parent eval-budget invariant from the
v3 spec is preserved exactly. Operators who need to observe retry
rates should read the loguru log: every retry emits a WARNING with
the attempt number and the parse error.
"""

from __future__ import annotations

from loguru import logger

from arid_badger.typing_utils import implements

from .domain import (
    AsyncSpeedupEstimator,
    KernelRuntimeEstimate,
    KernelRuntimeQuery,
    LlmCallUsage,
)
from .parsing import EstimatorParseError


class RetryingSpeedupEstimator:
    """Retries the inner estimator on :class:`EstimatorParseError`.

    The returned ``LlmCallUsage`` reflects only the successful
    attempt — tokens spent on failed attempts are not summed in.
    Inner estimators that don't report usage (return ``None``) keep
    that contract through the wrapper.
    """

    def __init__(
        self,
        inner: AsyncSpeedupEstimator,
        *,
        max_retries: int = 1,
    ) -> None:
        if max_retries < 0:
            raise ValueError(
                f"max_retries must be >= 0, got {max_retries}"
            )
        self._inner = inner
        self._max_retries = max_retries

    async def aestimate(
        self, query: KernelRuntimeQuery
    ) -> tuple[KernelRuntimeEstimate, LlmCallUsage | None]:
        last_error: EstimatorParseError | None = None
        # max_retries=N means N+1 total attempts (1 initial + N retries).
        for attempt in range(self._max_retries + 1):
            try:
                return await self._inner.aestimate(query)
            except EstimatorParseError as exc:
                last_error = exc
                attempts_remaining = self._max_retries - attempt
                if attempts_remaining > 0:
                    logger.warning(
                        "RetryingSpeedupEstimator: attempt {a}/{total} failed "
                        "with EstimatorParseError ({err}); retrying "
                        "({remaining} attempts remaining)",
                        a=attempt + 1,
                        total=self._max_retries + 1,
                        err=exc,
                        remaining=attempts_remaining,
                    )

        # All attempts exhausted — re-raise the most recent parse error.
        # ``last_error`` is guaranteed non-None here: the loop entered at
        # least once, and the only way out without returning is through
        # the except branch.
        assert last_error is not None
        raise last_error


implements(AsyncSpeedupEstimator)(RetryingSpeedupEstimator)
