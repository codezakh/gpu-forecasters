from __future__ import annotations

import asyncio
from pathlib import Path

from gpu_forecasters.landscape_map.v1.domain import (
    KernelImplementation,
    KernelRuntimeEstimate,
    KernelRuntimeQuery,
    KernelTaskInfo,
    LikertConfidence,
    LlmCallUsage,
    SpeedupBin,
)
from gpu_forecasters.landscape_map.v1.ranking import (
    ComparisonCacheEntry,
    aquery_pair,
    aswiss_tournament,
    comparison_cache_key,
    make_comparison_cache,
)


def _estimate(bin_: SpeedupBin) -> KernelRuntimeEstimate:
    return KernelRuntimeEstimate(
        predicted_bin=bin_,
        bin_confidences={bin_: LikertConfidence.HIGH},
        reasoning="",
    )


class _ScriptedEstimator:
    """Fake estimator: returns a scripted bin per (ref_name, cand_name)
    pair and records calls so tests can assert on cache reuse.
    """

    def __init__(self, script: dict[tuple[str, str], SpeedupBin]) -> None:
        self._script: dict[tuple[str, str], SpeedupBin] = script
        self.calls: list[tuple[str, str]] = []

    async def aestimate(
        self, query: KernelRuntimeQuery
    ) -> tuple[KernelRuntimeEstimate, LlmCallUsage | None]:
        pair = (query.reference.kernel_name, query.candidate.kernel_name)
        self.calls.append(pair)
        return _estimate(self._script[pair]), LlmCallUsage(input_tokens=1, output_tokens=1)


class _ExplodingEstimator:
    async def aestimate(
        self, query: KernelRuntimeQuery
    ) -> tuple[KernelRuntimeEstimate, LlmCallUsage | None]:
        del query
        raise RuntimeError("boom")


_TASK = KernelTaskInfo(op_name="toy", level_id=1, task_id=1)


def _impl(name: str) -> KernelImplementation:
    return KernelImplementation(kernel_name=name, code=f"# {name}", runtime_ms=None)


def _query(ref: str, cand: str) -> KernelRuntimeQuery:
    return KernelRuntimeQuery(task=_TASK, reference=_impl(ref), candidate=_impl(cand))


def test_comparison_cache_key_is_derived_from_query() -> None:
    key = comparison_cache_key(_query("alpha", "beta"))
    assert key == "L1_1_toy/alpha__beta"


def test_aquery_pair_miss_then_hit(tmp_path: Path) -> None:
    cache = make_comparison_cache(tmp_path)
    estimator = _ScriptedEstimator({("a", "b"): SpeedupBin.HIGH_SPEEDUP})

    async def scenario() -> None:
        sem = asyncio.Semaphore(1)
        first = await aquery_pair(
            query=_query("a", "b"), estimator=estimator, cache=cache, semaphore=sem,
        )
        assert first.is_ok()
        assert first.unwrap().from_cache is False

        second = await aquery_pair(
            query=_query("a", "b"), estimator=estimator, cache=cache, semaphore=sem,
        )
        assert second.is_ok()
        assert second.unwrap().from_cache is True

    asyncio.run(scenario())
    assert estimator.calls == [("a", "b")]  # second call hit cache


def test_aquery_pair_persists_only_value_fields(tmp_path: Path) -> None:
    cache = make_comparison_cache(tmp_path)
    estimator = _ScriptedEstimator({("a", "b"): SpeedupBin.MINOR_SPEEDUP})

    async def scenario() -> None:
        sem = asyncio.Semaphore(1)
        _ = await aquery_pair(
            query=_query("a", "b"), estimator=estimator, cache=cache, semaphore=sem,
        )

    asyncio.run(scenario())

    stored = cache.get(comparison_cache_key(_query("a", "b")))
    assert isinstance(stored, ComparisonCacheEntry)
    assert stored.estimate.predicted_bin == SpeedupBin.MINOR_SPEEDUP
    # Entry holds only the estimator output — no addressing metadata.
    assert set(ComparisonCacheEntry.model_fields.keys()) == {"estimate", "llm_usage"}


def test_aquery_pair_wraps_estimator_error(tmp_path: Path) -> None:
    cache = make_comparison_cache(tmp_path)

    async def scenario() -> None:
        sem = asyncio.Semaphore(1)
        result = await aquery_pair(
            query=_query("a", "b"),
            estimator=_ExplodingEstimator(),
            cache=cache,
            semaphore=sem,
        )
        assert result.is_err()
        err = result.unwrap_err()
        assert err.error_type == "RuntimeError"
        assert err.error_message == "boom"
        assert err.reference_id == "a"
        assert err.candidate_id == "b"

    asyncio.run(scenario())
    # Failed calls must not pollute the cache.
    assert cache.get(comparison_cache_key(_query("a", "b"))) is None


def test_aswiss_tournament_awards_wins_by_predicted_bin(tmp_path: Path) -> None:
    cache = make_comparison_cache(tmp_path)
    # Kernels: a, b, c, d. Round 0 order (all score 0, name-sorted): a, b, c, d.
    # Round-0 pairings: (a=ref, b=cand), (c=ref, d=cand).
    # Script: b beats a (MINOR_SPEEDUP → candidate win),
    #         c beats d (MINOR_SLOWDOWN → reference win).
    # After round 0: b=1, c=1, a=0, d=0.
    # Round 1 order: (b, c, a, d) → pairs (b=ref, c=cand), (a=ref, d=cand).
    # Script: c beats b, a beats d. Final: c=2, b=1, a=1, d=0.
    script = {
        ("a", "b"): SpeedupBin.MINOR_SPEEDUP,
        ("c", "d"): SpeedupBin.MINOR_SLOWDOWN,
        ("b", "c"): SpeedupBin.MINOR_SPEEDUP,
        ("a", "d"): SpeedupBin.MINOR_SLOWDOWN,
    }
    estimator = _ScriptedEstimator(script)
    kernels = [_impl("a"), _impl("b"), _impl("c"), _impl("d")]

    scores = asyncio.run(
        aswiss_tournament(
            task=_TASK,
            kernels=kernels,
            estimator=estimator,
            cache=cache,
            num_rounds=2,
            max_concurrency=4,
        )
    )

    assert scores == {"a": 1, "b": 1, "c": 2, "d": 0}


def test_aswiss_tournament_reruns_hit_cache(tmp_path: Path) -> None:
    cache = make_comparison_cache(tmp_path)
    script = {
        ("a", "b"): SpeedupBin.MINOR_SPEEDUP,
        ("c", "d"): SpeedupBin.MINOR_SLOWDOWN,
        ("b", "c"): SpeedupBin.MINOR_SPEEDUP,
        ("a", "d"): SpeedupBin.MINOR_SLOWDOWN,
    }
    estimator = _ScriptedEstimator(script)
    kernels = [_impl("a"), _impl("b"), _impl("c"), _impl("d")]

    first = asyncio.run(
        aswiss_tournament(
            task=_TASK, kernels=kernels, estimator=estimator, cache=cache,
            num_rounds=2, max_concurrency=4,
        )
    )
    calls_after_first = list(estimator.calls)

    second = asyncio.run(
        aswiss_tournament(
            task=_TASK, kernels=kernels, estimator=estimator, cache=cache,
            num_rounds=2, max_concurrency=4,
        )
    )

    assert first == second
    # Second run must produce zero new estimator calls.
    assert estimator.calls == calls_after_first
