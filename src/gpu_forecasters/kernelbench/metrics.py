"""KernelBench search-result metrics.

Canonical library home for summary metrics computed over a collection of
KernelBench evaluation results. Callers pass in raw rewards so the metric is
decoupled from any particular search algorithm's archive/checkpoint type.

By convention in the Modal evaluator, ``reward`` is the measured speedup for
valid kernels and ``None`` for failures.
"""

from __future__ import annotations

from collections.abc import Iterable


def compute_fast1_score(rewards: Iterable[float | None]) -> float:
    """Fraction of results that are correct with strictly greater than 1x speedup.

    A result counts toward fast_1 iff its reward is not None and > 1.0.
    Returns 0.0 for an empty collection.
    """
    rewards_list = list(rewards)
    total = len(rewards_list)
    if total == 0:
        return 0.0
    fast1_count = sum(1 for r in rewards_list if r is not None and r > 1.0)
    return fast1_count / total
