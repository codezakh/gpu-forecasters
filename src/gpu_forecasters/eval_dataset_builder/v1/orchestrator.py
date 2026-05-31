"""Top-level orchestrator: harvest → fill shortfalls → write JSONL + manifest.

The pipeline is built around three small helpers (``harvest_into_eval_set``,
``fill_via_generation``, ``write_eval_set``) plus one entry point
(``build_eval_dataset``) that composes them. Pulling the helpers out
keeps the entry point short and lets future callers reuse harvest/write
without the full orchestrator.

The library does not own a checkpoint format. Harvest is decoupled via
the ``HarvestedKernelSource`` protocol — the caller ships an adapter
that yields ``KernelRuntimeComparison`` rows from whatever shape their
prior search produced.
"""

from __future__ import annotations

import hashlib
import logging
from collections import defaultdict
from concurrent.futures import as_completed
from datetime import datetime, timezone
from itertools import islice
from pathlib import Path
from collections.abc import Iterable
from typing import Protocol

from gpu_forecasters.gpu_mode_kernel.core import CaseSpeedupT, GpuModeKernelObservation
from gpu_forecasters.gpu_mode_kernel.kernel_pack import TestArgsT
from gpu_forecasters.gpu_mode_kernel.modal_scoring import PackedModalRuntime
from gpu_forecasters.landscape_map.v1.domain import HardwareContext, SpeedupBin
from gpu_forecasters.max_reward_puct.v2.config import SearchConfig
from gpu_forecasters.max_reward_puct.v2.providers import AsyncEvaluationProvider

from .bin_filler import BinFiller
from .domain import (
    BinFillRequest,
    EvalDataset,
    EvaluationProviderSpec,
    EvalSet,
    EvalSetManifest,
    HarvestedKernelSource,
    KernelGenerationAttempt,
    KernelRuntimeComparison,
    MutationProviderSpec,
    NumKernelsForSpeedupBin,
    RequestForKernelInGoalSpeedupBin,
)


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Harvest.
# ---------------------------------------------------------------------------


def harvest_into_eval_set(
    rows: Iterable[KernelRuntimeComparison],
    *,
    eval_set: EvalSet | None = None,
) -> EvalSet:
    """Group harvested kernels into an ``EvalSet`` keyed by ``true_bin``.

    Returns a new dict; the input ``eval_set`` (if provided) is not
    mutated. Existing rows are preserved and harvested rows are
    appended to their respective bin lists.
    """
    out: defaultdict[SpeedupBin, list[KernelRuntimeComparison]] = defaultdict(list)
    if eval_set is not None:
        for bin_, items in eval_set.items():
            out[bin_].extend(items)
    for row in rows:
        out[row.true_bin].append(row)
    return dict(out)


# ---------------------------------------------------------------------------
# Fill via a generic generator + evaluator (kept for the orchestrator's
# unit-test seam — exercises the same shape ``BinFiller`` does without
# requiring Modal). ``BinFiller`` is the production fill path.
# ---------------------------------------------------------------------------


def fill_via_generation(
    eval_set: EvalSet,
    *,
    target: NumKernelsForSpeedupBin,
    reference_code: str,
    hardware: HardwareContext,
    generator: "_KernelGenerator",
    evaluator: AsyncEvaluationProvider[GpuModeKernelObservation[CaseSpeedupT]],
    max_attempts_per_bin: int,
) -> tuple[EvalSet, list[KernelGenerationAttempt[CaseSpeedupT]]]:
    """Fill bins short of their target by pulling candidates and evaluating them.

    For each bin in ``target``: if ``eval_set`` already meets the target,
    skip. Otherwise pull up to ``max_attempts_per_bin`` candidates from
    ``generator``, submit them all to the evaluator, gather results in
    completion order via ``as_completed``, accept iff the measured bin
    equals the target bin, log every attempt either way, and drain all
    submitted futures even after the bin fills.

    Returns the updated eval set plus the full attempt log across all bins.
    The input eval set is not mutated.
    """
    out: dict[SpeedupBin, list[KernelRuntimeComparison]] = {
        b: list(items) for b, items in eval_set.items()
    }
    attempts: list[KernelGenerationAttempt[CaseSpeedupT]] = []

    for target_bin, n_target in target.items():
        already_have = out.get(target_bin, [])
        needed = n_target - len(already_have)
        if needed <= 0:
            continue

        request = RequestForKernelInGoalSpeedupBin(
            target_bin=target_bin,
            reference_code=reference_code,
            hardware=hardware,
        )
        candidates = list(islice(generator.generate(request), max_attempts_per_bin))
        if not candidates:
            continue

        future_to_code = {evaluator.submit(code): code for code in candidates}
        accepted: list[KernelRuntimeComparison] = []

        for future in as_completed(future_to_code):
            candidate_code = future_to_code[future]
            evaluation = future.result()
            attempts.append(
                KernelGenerationAttempt[CaseSpeedupT](
                    request=request,
                    candidate_code=candidate_code,
                    evaluation=evaluation,
                )
            )

            if len(accepted) >= needed:
                continue
            feedback = evaluation.observation.feedback
            if feedback.kind != "success":
                continue
            speedup = feedback.aggregated_speedup
            measured_bin = SpeedupBin.from_speedup(speedup)
            if measured_bin != target_bin:
                continue
            accepted.append(
                KernelRuntimeComparison(
                    reference_code=reference_code,
                    candidate_code=candidate_code,
                    hardware=hardware,
                    aggregated_speedup=speedup,
                    true_bin=measured_bin,
                    source_id=_generated_source_id(candidate_code),
                )
            )

        out.setdefault(target_bin, []).extend(accepted)

    return out, attempts


def _generated_source_id(candidate_code: str) -> str:
    digest = hashlib.sha256(candidate_code.encode("utf-8")).hexdigest()[:12]
    return f"generated/{digest}"


# Local Protocol — used by ``fill_via_generation`` only. Production fill
# goes through ``BinFiller`` (which has its own goal-conditioned generator
# baked in).
class _KernelGenerator(Protocol):
    def generate(self, request: RequestForKernelInGoalSpeedupBin) -> Iterable[str]: ...


# ---------------------------------------------------------------------------
# Write.
# ---------------------------------------------------------------------------


def write_eval_set(
    output_dir: Path,
    eval_set: EvalSet,
    manifest: EvalSetManifest,
) -> tuple[Path, Path]:
    """Write the eval set's JSONL and manifest atomically.

    Returns ``(jsonl_path, manifest_path)``. The JSONL is a flat list of
    ``KernelRuntimeComparison`` rows in bin order; the manifest is a
    single JSON object.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    jsonl_path = output_dir / "eval_dataset.jsonl"
    jsonl_tmp = jsonl_path.with_suffix(jsonl_path.suffix + ".tmp")
    with jsonl_tmp.open("w") as f:
        for bin_ in sorted(eval_set.keys(), key=int):
            for comparison in eval_set[bin_]:
                _ = f.write(comparison.model_dump_json() + "\n")
    _ = jsonl_tmp.replace(jsonl_path)

    manifest_path = output_dir / "eval_dataset_manifest.json"
    manifest_tmp = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    _ = manifest_tmp.write_text(manifest.model_dump_json(indent=2))
    _ = manifest_tmp.replace(manifest_path)

    return jsonl_path, manifest_path


def read_eval_dataset(output_dir: Path) -> EvalDataset:
    """Read the on-disk artifact ``write_eval_set`` produced.

    Reciprocal of ``write_eval_set``: parses ``eval_dataset.jsonl`` +
    ``eval_dataset_manifest.json`` from ``output_dir`` and returns the
    loaded value. Raises if either file is missing or malformed —
    eval datasets are an internal contract, not user input.
    """
    jsonl_path = output_dir / "eval_dataset.jsonl"
    manifest_path = output_dir / "eval_dataset_manifest.json"

    comparisons = [
        KernelRuntimeComparison.model_validate_json(line)
        for line in jsonl_path.read_text().splitlines()
        if line.strip()
    ]
    manifest = EvalSetManifest.model_validate_json(manifest_path.read_text())
    return EvalDataset(comparisons=comparisons, manifest=manifest)


# ---------------------------------------------------------------------------
# Orchestrator entry point.
# ---------------------------------------------------------------------------


def build_eval_dataset(
    *,
    pack_runtime: PackedModalRuntime[TestArgsT, CaseSpeedupT],
    harvested_source: HarvestedKernelSource,
    target: NumKernelsForSpeedupBin,
    reference_code: str,
    hardware: HardwareContext,
    output_dir: Path,
    search: SearchConfig,
    mutation: MutationProviderSpec,
    evaluation: EvaluationProviderSpec,
    source_search_tag: str,
    invocation_sink_dir: Path | None = None,
) -> tuple[Path, Path]:
    """Harvest, fill bin shortfalls via goal-directed PUCT, write JSONL + manifest.

    Returns ``(jsonl_path, manifest_path)``.

    Steps:

    1. Pull harvested rows by calling ``harvested_source()``; group by
       ``true_bin`` into an ``EvalSet``.
    2. For each bin in ``target`` whose ``EvalSet`` count is below the
       target, run one ``BinFiller.fill(...)`` per shortfall bin (in
       ``SpeedupBin`` order). Append produced ``in_target_kernels`` to
       the eval set, capped at ``target[bin]``.
    3. Write JSONL + manifest atomically.

    The Modal session is opened once for the whole run via
    ``BinFiller.__enter__`` and shared across all per-bin fills.

    The ``output_dir`` will receive ``eval_dataset.jsonl``,
    ``eval_dataset_manifest.json``, and one subdirectory per filled bin
    containing that bin's durable artifacts (``events.jsonl``,
    ``summary.json``, ``seed.json``, ``sample_prompt.txt``).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Harvest.
    eval_set = harvest_into_eval_set(harvested_source())
    harvested_per_bin = {bin_: len(items) for bin_, items in eval_set.items()}
    total_harvested = sum(harvested_per_bin.values())
    logger.info(
        "Harvest done: %d kernels across %d bins (%s)",
        total_harvested,
        len(harvested_per_bin),
        ", ".join(
            f"bin{int(b)}={n}"
            for b, n in sorted(harvested_per_bin.items(), key=lambda kv: int(kv[0]))
        ),
    )

    # 2. Fill shortfalls.
    generated_per_bin: dict[SpeedupBin, int] = {}
    attempts_per_bin: dict[SpeedupBin, int] = {}
    shortfall_bins = sorted(
        (b for b in target if len(eval_set.get(b, [])) < target[b]),
        key=int,
    )
    if shortfall_bins:
        with BinFiller[TestArgsT, CaseSpeedupT](
            pack_runtime=pack_runtime,
            reference_code=reference_code,
            hardware=hardware,
            search=search,
            mutation=mutation,
            evaluation=evaluation,
            invocation_sink_dir=invocation_sink_dir,
        ) as filler:
            for target_bin in shortfall_bins:
                needed = target[target_bin] - len(eval_set.get(target_bin, []))
                # Flatten the harvest pool — the seed picker runs over
                # all harvested kernels, not just those in the target
                # bin, because the closest-to-midpoint heuristic may
                # prefer a neighboring-bin kernel.
                harvested_pool = [row for items in eval_set.values() for row in items]
                bin_dir = output_dir / target_bin.name
                result = filler.fill(
                    BinFillRequest(
                        target_bin=target_bin,
                        harvested=harvested_pool,
                        output_dir=bin_dir,
                    )
                )
                eval_set.setdefault(target_bin, []).extend(
                    result.in_target_kernels[:needed]
                )
                generated_per_bin[target_bin] = min(
                    needed, len(result.in_target_kernels)
                )
                attempts_per_bin[target_bin] = result.summary.total_candidates_evaluated

    # 3. Write.
    manifest = EvalSetManifest(
        source_search_tag=source_search_tag,
        hardware=hardware,
        harvested_per_bin=harvested_per_bin,
        generated_per_bin=generated_per_bin,
        attempts_per_bin=attempts_per_bin,
        generated_at=datetime.now(timezone.utc),
    )
    jsonl_path, manifest_path = write_eval_set(output_dir, eval_set, manifest)
    logger.info("Wrote eval set: %s", jsonl_path)
    logger.info("Wrote manifest: %s", manifest_path)

    shortfalls = {
        bin_: target[bin_] - len(eval_set.get(bin_, []))
        for bin_ in target
        if len(eval_set.get(bin_, [])) < target[bin_]
    }
    if shortfalls:
        logger.info(
            "Bins still short of target after fill: %s",
            ", ".join(
                f"bin{int(b)} needs {n} more"
                for b, n in sorted(shortfalls.items(), key=lambda kv: int(kv[0]))
            ),
        )

    return jsonl_path, manifest_path
