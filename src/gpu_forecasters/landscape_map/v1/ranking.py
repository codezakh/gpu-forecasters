"""Cached pairwise ranking over an `AsyncSpeedupEstimator`.

The unit of work is a single pairwise comparison, expressed as a
`KernelRuntimeQuery` — which already carries everything identity-related
(task info, reference impl, candidate impl). The cache entry stores only
what the estimator produced; the filesystem key is a pure function of
the query, so the addressing scheme can't drift from the stored value.

The module exposes:

- `aquery_pair`: durable, semaphore-bounded wrapper around `aestimate`.
- `aswiss_tournament`: adaptive Swiss-system tournament built on top.
- `comparison_cache_key` + `make_comparison_cache`: helpers so every
  call site (tournament, greedy-vs-reference selection, etc.) derives
  the same key from the same query.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from loguru import logger
from pydantic import BaseModel

from gpu_forecasters.cache import FileCache
from gpu_forecasters.landscape_map.v1.domain import (
    AsyncSpeedupEstimator,
    KernelImplementation,
    KernelRuntimeEstimate,
    KernelRuntimeQuery,
    KernelTaskInfo,
    LlmCallUsage,
    SpeedupBin,
)
from gpu_forecasters.typing_utils import Err, Ok, Option


class ComparisonCacheEntry(BaseModel, frozen=True):
    """Stored value of a cached pairwise estimator call.

    Holds only what `aestimate` returned. The addressing metadata (task,
    reference, candidate) lives in the filesystem path, derived from the
    originating `KernelRuntimeQuery` via `comparison_cache_key`.
    """

    estimate: KernelRuntimeEstimate
    llm_usage: LlmCallUsage | None


class ComparisonRecord(BaseModel, frozen=True):
    """In-memory result of `aquery_pair`. Includes a `from_cache` flag so
    callers can distinguish cold calls from hits for logging / accounting.
    """

    estimate: KernelRuntimeEstimate
    llm_usage: LlmCallUsage | None
    from_cache: bool


class ComparisonError(BaseModel, frozen=True):
    """Estimator failure wrapped for aggregation inside `asyncio.gather`.

    Carries the reference/candidate/task identifiers purely for log
    messages — nothing reads them back for routing.
    """

    task_op_name: str
    reference_id: str
    candidate_id: str
    error_type: str
    error_message: str


def comparison_cache_key(query: KernelRuntimeQuery) -> str:
    """Pure: the `FileCache` key at which a comparison's entry lives.

    The layout is `L{level}_{task}_{op}/{ref}__{cand}.json`. Every
    `FileCache.get` and `FileCache.put` call site in the ranking module
    goes through this helper so the key has exactly one source of truth.
    """
    task = query.task
    return (
        f"L{task.level_id}_{task.task_id}_{task.op_name}"
        f"/{query.reference.kernel_name}__{query.candidate.kernel_name}"
    )


def make_comparison_cache(root: Path) -> FileCache[ComparisonCacheEntry]:
    """Construct a `FileCache[ComparisonCacheEntry]` at `root/comparisons/`."""
    return FileCache(
        root=root / "comparisons",
        value_type=ComparisonCacheEntry,
    )


async def aquery_pair(
    *,
    query: KernelRuntimeQuery,
    estimator: AsyncSpeedupEstimator,
    cache: FileCache[ComparisonCacheEntry],
    semaphore: asyncio.Semaphore,
) -> Option[ComparisonRecord, ComparisonError]:
    """Cached, semaphore-bounded single pairwise comparison.

    On cache hit returns immediately with `from_cache=True` and never
    acquires the semaphore. On miss: acquires the semaphore, calls the
    estimator, writes the result to the cache atomically, and returns.
    Estimator failures are wrapped in `ComparisonError` rather than
    raised, so `asyncio.gather` over many calls never fails whole-cloth.
    """
    key = comparison_cache_key(query)
    hit = cache.get(key)
    if hit is not None:
        return Ok(
            ComparisonRecord(
                estimate=hit.estimate,
                llm_usage=hit.llm_usage,
                from_cache=True,
            )
        )

    async with semaphore:
        try:
            estimate, llm_usage = await estimator.aestimate(query)
        except Exception as exc:
            logger.error(
                "aquery_pair failed for {ref} vs {cand} (op={op}): {err}",
                ref=query.reference.kernel_name,
                cand=query.candidate.kernel_name,
                op=query.task.op_name,
                err=exc,
            )
            return Err(
                ComparisonError(
                    task_op_name=query.task.op_name,
                    reference_id=query.reference.kernel_name,
                    candidate_id=query.candidate.kernel_name,
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                )
            )

    cache.put(key, ComparisonCacheEntry(estimate=estimate, llm_usage=llm_usage))
    return Ok(
        ComparisonRecord(estimate=estimate, llm_usage=llm_usage, from_cache=False)
    )


async def aswiss_tournament(
    *,
    task: KernelTaskInfo,
    kernels: list[KernelImplementation],
    estimator: AsyncSpeedupEstimator,
    cache: FileCache[ComparisonCacheEntry],
    num_rounds: int,
    max_concurrency: int,
) -> dict[str, int]:
    """Swiss-system tournament over `kernels`, returning `kernel_name -> wins`.

    Each round: sort by current score descending (ties broken by name
    for determinism), pair adjacent kernels, query the estimator for
    every pair in parallel under a semaphore. A predicted bin of
    `MINOR_SPEEDUP` or higher means the candidate (right side of the
    pair) won; `MINOR_SLOWDOWN` or lower means the reference (left side)
    won; the `~1.0x` boundary is a tie (no points). On an odd number of
    kernels the lowest-ranked one sits the round out, same as standard
    Swiss.

    Every pairwise result goes through `aquery_pair`, so a crash mid-run
    loses no already-completed comparisons: a rerun re-enters round 0,
    hits cache for every previously-seen pair, and only spends LLM calls
    on new pairings. Duplicate `kernel_name` values are the caller's
    problem — the tournament treats them as distinct entries but the
    cache will collapse their pairings.
    """
    scores: dict[str, int] = {k.kernel_name: 0 for k in kernels}
    by_name: dict[str, KernelImplementation] = {k.kernel_name: k for k in kernels}
    semaphore = asyncio.Semaphore(max_concurrency)

    for _round in range(num_rounds):
        current_order = sorted(scores.keys(), key=lambda name: (-scores[name], name))
        pairs = [
            (current_order[i], current_order[i + 1])
            for i in range(0, len(current_order) - 1, 2)
        ]

        results = await asyncio.gather(
            *[
                aquery_pair(
                    query=KernelRuntimeQuery(
                        task=task,
                        reference=by_name[ref_name],
                        candidate=by_name[cand_name],
                    ),
                    estimator=estimator,
                    cache=cache,
                    semaphore=semaphore,
                )
                for ref_name, cand_name in pairs
            ],
            return_exceptions=False,
        )

        for (ref_name, cand_name), result in zip(pairs, results):
            if result.is_ok():
                predicted_bin = result.unwrap().estimate.predicted_bin
                if predicted_bin >= SpeedupBin.MINOR_SPEEDUP:
                    scores[cand_name] += 1
                elif predicted_bin <= SpeedupBin.MINOR_SLOWDOWN:
                    scores[ref_name] += 1
            else:
                error = result.unwrap_err()
                logger.warning(
                    "aswiss_tournament: pair failed {ref} vs {cand}: {err}",
                    ref=error.reference_id,
                    cand=error.candidate_id,
                    err=error.error_message,
                )

    return scores
