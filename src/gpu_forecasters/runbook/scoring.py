"""Async surrogate-scoring loop shared by 01, 03, and 05.

Lifts the per-row scoring path out of the individual scripts so the
JSON-config → typed-row → forecast → JSONL flow lives in one place.
The same estimator handles the canonical eval set and the discovery
pairs — only the dataset loader differs.
"""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Sequence

from loguru import logger
from pydantic import BaseModel

from gpu_forecasters.cache import FileCache
from gpu_forecasters.eval_dataset_builder.v1 import KernelRuntimeComparison
from gpu_forecasters.landscape_map.v2 import (
    AsyncSpeedupEstimator,
    HardwareContext,
    KernelImplementation,
    KernelRuntimeEstimate,
    KernelRuntimeQuery,
    KernelTaskInfo,
    LlmCallUsage,
)


class ScoredComparison(BaseModel, frozen=True):
    """One scored row from one repeat.

    ``prediction`` is ``None`` on a parse failure; ``parse_error`` then
    carries the exception's ``type: msg``. Cached under
    ``repeat_<i>/<pack>/<source_id>`` so concurrent repeats write to
    disjoint files and a crash mid-run loses no LLM calls.
    """

    pack_name: str
    source_id: str
    comparison: KernelRuntimeComparison
    repeat: int
    prediction: KernelRuntimeEstimate | None
    parse_error: str | None
    llm_usage: LlmCallUsage | None
    elapsed_s: float


def _build_query(
    *, pack_name: str, comparison: KernelRuntimeComparison
) -> KernelRuntimeQuery:
    hardware = HardwareContext(**comparison.hardware.model_dump())
    return KernelRuntimeQuery(
        task=KernelTaskInfo(op_name=pack_name, level_id=0, task_id=0),
        reference=KernelImplementation(
            kernel_name="reference", code=comparison.reference_code, runtime_ms=None
        ),
        candidate=KernelImplementation(
            kernel_name="candidate", code=comparison.candidate_code, runtime_ms=None
        ),
        hardware=hardware,
    )


async def _score_one(
    *,
    pack_name: str,
    comparison: KernelRuntimeComparison,
    repeat: int,
    estimator: AsyncSpeedupEstimator,
    cache: FileCache[ScoredComparison],
    semaphore: asyncio.Semaphore,
) -> ScoredComparison:
    key = f"repeat_{repeat}/{pack_name}/{comparison.source_id}"
    hit = cache.get(key)
    if hit is not None:
        return hit

    async with semaphore:
        start = time.monotonic()
        query = _build_query(pack_name=pack_name, comparison=comparison)
        try:
            estimate, usage = await estimator.aestimate(query)
            scored = ScoredComparison(
                pack_name=pack_name,
                source_id=comparison.source_id,
                comparison=comparison,
                repeat=repeat,
                prediction=estimate,
                parse_error=None,
                llm_usage=usage,
                elapsed_s=time.monotonic() - start,
            )
        except Exception as exc:
            logger.warning("scoring failed for {key}: {err}", key=key, err=exc)
            scored = ScoredComparison(
                pack_name=pack_name,
                source_id=comparison.source_id,
                comparison=comparison,
                repeat=repeat,
                prediction=None,
                parse_error=f"{type(exc).__name__}: {exc}",
                llm_usage=None,
                elapsed_s=time.monotonic() - start,
            )

    cache.put(key, scored)
    return scored


async def ascore_pack_repeat(
    *,
    pack_name: str,
    comparisons: Sequence[KernelRuntimeComparison],
    repeat: int,
    estimator: AsyncSpeedupEstimator,
    cache: FileCache[ScoredComparison],
    semaphore: asyncio.Semaphore,
) -> list[ScoredComparison]:
    """Score every comparison in one (pack, repeat) cell concurrently."""
    return await asyncio.gather(
        *(
            _score_one(
                pack_name=pack_name,
                comparison=c,
                repeat=repeat,
                estimator=estimator,
                cache=cache,
                semaphore=semaphore,
            )
            for c in comparisons
        )
    )


def write_jsonl(path: Path, items: Sequence[ScoredComparison]) -> None:
    """Atomically write a list of scored rows to ``path``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        for it in items:
            _ = f.write(it.model_dump_json() + "\n")
    _ = tmp.replace(path)


def summarize_repeat(
    *, pack_name: str, repeat: int, scored: Sequence[ScoredComparison]
) -> dict[str, object]:
    """Compute the on-the-fly summary the runner prints after each cell.

    Cheap metrics only — full calibration is a separate analysis. The
    summary is for "is this run going well" rather than for paper-grade
    numbers.
    """
    n = len(scored)
    parsed = [s for s in scored if s.prediction is not None]
    n_parsed = len(parsed)

    def _truth_bin(s: ScoredComparison) -> int:
        return int(s.comparison.true_bin)

    exact = sum(
        1
        for s in parsed
        if s.prediction is not None and int(s.prediction.predicted_bin) == _truth_bin(s)
    )
    within_one = sum(
        1
        for s in parsed
        if s.prediction is not None
        and abs(int(s.prediction.predicted_bin) - _truth_bin(s)) <= 1
    )
    elapsed = [s.elapsed_s for s in scored]
    mean_elapsed = sum(elapsed) / n if n > 0 else 0.0
    return {
        "pack_name": pack_name,
        "repeat": repeat,
        "n_total": n,
        "n_parsed": n_parsed,
        "n_failed": n - n_parsed,
        "exact_match": exact,
        "off_by_one_or_less": within_one,
        "mean_elapsed_s": mean_elapsed,
    }


def write_index(
    *,
    output_dir: Path,
    surrogate_label: str,
    config_snapshot: BaseModel,
    summaries: Sequence[dict[str, object]],
) -> None:
    payload = {
        "surrogate_label": surrogate_label,
        "config": json.loads(config_snapshot.model_dump_json()),
        "summaries": list(summaries),
    }
    (output_dir / "index.json").write_text(json.dumps(payload, indent=2))


__all__ = [
    "ScoredComparison",
    "ascore_pack_repeat",
    "summarize_repeat",
    "write_index",
    "write_jsonl",
]
