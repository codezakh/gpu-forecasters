"""Orchestrator tests: harvest, fill loop, and write.

The full ``build_eval_dataset`` entry point requires Modal and is
covered by the smoke-test experiment, not here. These tests exercise
the three composable helpers — ``harvest_into_eval_set``,
``fill_via_generation``, and ``write_eval_set`` — over fakes, plus the
acceptance/rejection logic that ``BinFiller`` would otherwise hide.
"""

from __future__ import annotations

import json
from concurrent.futures import Future
from datetime import datetime, timezone
from pathlib import Path
from typing import Self, final

from gpu_forecasters.eval_dataset_builder.v1.domain import (
    EvalDataset,
    EvalSet,
    EvalSetManifest,
    KernelRuntimeComparison,
    RequestForKernelInGoalSpeedupBin,
)
from gpu_forecasters.eval_dataset_builder.v1.orchestrator import (
    fill_via_generation,
    harvest_into_eval_set,
    read_eval_dataset,
    write_eval_set,
)
from gpu_forecasters.gpu_mode_kernel.core import (
    CompileFailedFeedback,
    GpuModeKernelObservation,
    SuccessFeedback,
)
from gpu_forecasters.gpu_mode_kernel.packs.trimul import TriMulCaseSpeedup
from gpu_forecasters.hill_climbing.domain import Evaluation
from gpu_forecasters.landscape_map.v1.domain import HardwareContext, SpeedupBin


_HARDWARE = HardwareContext(
    device_name="fake-gpu",
    compute_capability=(0, 0),
    total_global_memory_gb=0.0,
    multiprocessor_count=0,
    max_threads_per_multiprocessor=0,
    clock_rate_ghz=0.0,
    memory_clock_rate_ghz=0.0,
    memory_bus_width_bits=0,
)
_REFERENCE = "REF_CODE"


def _comparison(
    *, candidate_code: str, speedup: float, source_id: str
) -> KernelRuntimeComparison:
    return KernelRuntimeComparison(
        reference_code=_REFERENCE,
        candidate_code=candidate_code,
        hardware=_HARDWARE,
        aggregated_speedup=speedup,
        true_bin=SpeedupBin.from_speedup(speedup),
        source_id=source_id,
    )


def _success(speedup: float) -> Evaluation[GpuModeKernelObservation[TriMulCaseSpeedup]]:
    feedback: SuccessFeedback[TriMulCaseSpeedup] = SuccessFeedback(
        aggregated_speedup=speedup,
        aggregation_method="geomean",
        per_case_speedups=[],
    )
    observation = GpuModeKernelObservation[TriMulCaseSpeedup](feedback=feedback)
    return Evaluation[GpuModeKernelObservation[TriMulCaseSpeedup]](
        observation=observation, reward=speedup
    )


def _failure() -> Evaluation[GpuModeKernelObservation[TriMulCaseSpeedup]]:
    feedback = CompileFailedFeedback(compilation_error="boom")
    observation = GpuModeKernelObservation[TriMulCaseSpeedup](feedback=feedback)
    return Evaluation[GpuModeKernelObservation[TriMulCaseSpeedup]](
        observation=observation, reward=None
    )


def _resolved_future(
    value: Evaluation[GpuModeKernelObservation[TriMulCaseSpeedup]],
) -> Future[Evaluation[GpuModeKernelObservation[TriMulCaseSpeedup]]]:
    fut: Future[Evaluation[GpuModeKernelObservation[TriMulCaseSpeedup]]] = Future()
    fut.set_result(value)
    return fut


@final
class _ScriptedGenerator:
    def __init__(self, candidates: list[str]) -> None:
        self._candidates: list[str] = candidates

    def generate(self, request: RequestForKernelInGoalSpeedupBin) -> list[str]:  # pyright: ignore[reportUnusedParameter]
        return self._candidates


@final
class _ScriptedAsyncEvaluator:
    def __init__(
        self,
        outcomes: dict[str, Evaluation[GpuModeKernelObservation[TriMulCaseSpeedup]]],
    ) -> None:
        self._outcomes: dict[
            str, Evaluation[GpuModeKernelObservation[TriMulCaseSpeedup]]
        ] = outcomes

    def submit(
        self, program_code: str
    ) -> Future[Evaluation[GpuModeKernelObservation[TriMulCaseSpeedup]]]:
        return _resolved_future(self._outcomes[program_code])

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        return None


# --- harvest_into_eval_set --------------------------------------------------


def test_harvest_groups_by_true_bin() -> None:
    rows = [
        _comparison(candidate_code="A", speedup=2.5, source_id="s/a"),  # SIGNIFICANT
        _comparison(candidate_code="B", speedup=1.6, source_id="s/b"),  # MINOR
        _comparison(candidate_code="C", speedup=2.6, source_id="s/c"),  # SIGNIFICANT
    ]
    grouped = harvest_into_eval_set(rows)
    assert {b: [r.candidate_code for r in items] for b, items in grouped.items()} == {
        SpeedupBin.SIGNIFICANT_SPEEDUP: ["A", "C"],
        SpeedupBin.MINOR_SPEEDUP: ["B"],
    }


def test_harvest_appends_to_existing_bins() -> None:
    pre = _comparison(candidate_code="X", speedup=1.6, source_id="pre/x")
    rows = [_comparison(candidate_code="Y", speedup=1.7, source_id="s/y")]
    grouped = harvest_into_eval_set(rows, eval_set={SpeedupBin.MINOR_SPEEDUP: [pre]})
    assert [r.candidate_code for r in grouped[SpeedupBin.MINOR_SPEEDUP]] == ["X", "Y"]


# --- fill_via_generation ----------------------------------------------------


def test_fill_via_generation_fills_short_bin_and_skips_full_bin() -> None:
    pre_existing_in_bin_6 = _comparison(
        candidate_code="HARVESTED_BIN_6",
        speedup=2.2,  # SIGNIFICANT_SPEEDUP
        source_id="harvest/x",
    )
    eval_set: EvalSet = {SpeedupBin.SIGNIFICANT_SPEEDUP: [pre_existing_in_bin_6]}

    target = {
        SpeedupBin.MINOR_SPEEDUP: 2,
        SpeedupBin.SIGNIFICANT_SPEEDUP: 1,  # already satisfied by harvest
    }

    candidates = ["A_in_target", "B_off_target", "C_failure", "D_in_target", "E_unused"]
    outcomes = {
        "A_in_target": _success(1.6),  # bin 5 (MINOR_SPEEDUP)
        "B_off_target": _success(2.5),  # bin 6
        "C_failure": _failure(),
        "D_in_target": _success(1.7),  # bin 5
        "E_unused": _success(1.5),  # never submitted (capped at max_attempts_per_bin=4)
    }
    generator = _ScriptedGenerator(candidates)
    evaluator = _ScriptedAsyncEvaluator(outcomes)

    filled, attempts = fill_via_generation(
        eval_set,
        target=target,
        reference_code=_REFERENCE,
        hardware=_HARDWARE,
        generator=generator,
        evaluator=evaluator,
        max_attempts_per_bin=4,
    )

    # SIGNIFICANT_SPEEDUP was already at quota; harvest preserved unchanged.
    assert filled[SpeedupBin.SIGNIFICANT_SPEEDUP] == [pre_existing_in_bin_6]
    # MINOR_SPEEDUP filled to 2 by the two in-target successes.
    minor = filled[SpeedupBin.MINOR_SPEEDUP]
    assert {c.candidate_code for c in minor} == {"A_in_target", "D_in_target"}
    assert all(c.true_bin is SpeedupBin.MINOR_SPEEDUP for c in minor)
    # Four attempts logged: A, B, C, D — capped by max_attempts_per_bin=4.
    assert {a.candidate_code for a in attempts} == {
        "A_in_target",
        "B_off_target",
        "C_failure",
        "D_in_target",
    }
    assert all(a.request.target_bin is SpeedupBin.MINOR_SPEEDUP for a in attempts)


# --- write_eval_set ---------------------------------------------------------


def test_write_eval_set_round_trips(tmp_path: Path) -> None:
    eval_set: EvalSet = {
        SpeedupBin.MINOR_SPEEDUP: [
            _comparison(candidate_code="A", speedup=1.6, source_id="src/a"),
        ],
        SpeedupBin.SIGNIFICANT_SPEEDUP: [
            _comparison(candidate_code="B", speedup=2.0, source_id="src/b"),
        ],
    }
    manifest = EvalSetManifest(
        source_search_tag="test",
        hardware=_HARDWARE,
        harvested_per_bin={
            SpeedupBin.MINOR_SPEEDUP: 1,
            SpeedupBin.SIGNIFICANT_SPEEDUP: 1,
        },
        generated_per_bin={},
        attempts_per_bin={},
        generated_at=datetime(2026, 4, 29, tzinfo=timezone.utc),
    )

    jsonl_path, manifest_path = write_eval_set(tmp_path, eval_set, manifest)

    # JSONL: two rows in bin order (MINOR_SPEEDUP=5 first, SIGNIFICANT_SPEEDUP=6 second).
    lines = jsonl_path.read_text().strip().split("\n")
    assert len(lines) == 2
    parsed_codes = [json.loads(line)["candidate_code"] for line in lines]
    assert parsed_codes == ["A", "B"]

    # Manifest is valid JSON.
    manifest_data = json.loads(manifest_path.read_text())
    assert manifest_data["source_search_tag"] == "test"


# --- read_eval_dataset ------------------------------------------------------


def test_read_eval_dataset_round_trips_write_eval_set(tmp_path: Path) -> None:
    eval_set: EvalSet = {
        SpeedupBin.MINOR_SPEEDUP: [
            _comparison(candidate_code="A", speedup=1.6, source_id="src/a"),
        ],
        SpeedupBin.SIGNIFICANT_SPEEDUP: [
            _comparison(candidate_code="B", speedup=2.0, source_id="src/b"),
            _comparison(candidate_code="C", speedup=2.5, source_id="src/c"),
        ],
    }
    manifest = EvalSetManifest(
        source_search_tag="round-trip",
        hardware=_HARDWARE,
        harvested_per_bin={SpeedupBin.MINOR_SPEEDUP: 1, SpeedupBin.SIGNIFICANT_SPEEDUP: 2},
        generated_per_bin={},
        attempts_per_bin={},
        generated_at=datetime(2026, 4, 29, tzinfo=timezone.utc),
    )
    _ = write_eval_set(tmp_path, eval_set, manifest)

    loaded = read_eval_dataset(tmp_path)

    assert isinstance(loaded, EvalDataset)
    assert loaded.manifest == manifest
    assert [r.candidate_code for r in loaded.comparisons] == ["A", "B", "C"]
    # by_bin reconstructs the original grouping (modulo dict ordering).
    by_bin = loaded.by_bin()
    assert set(by_bin.keys()) == set(eval_set.keys())
    assert [r.candidate_code for r in by_bin[SpeedupBin.SIGNIFICANT_SPEEDUP]] == ["B", "C"]


def test_read_eval_dataset_skips_blank_lines(tmp_path: Path) -> None:
    # write_eval_set produces no blank lines, but tolerate them defensively
    # since the JSONL is line-oriented.
    jsonl = tmp_path / "eval_dataset.jsonl"
    row = _comparison(candidate_code="A", speedup=1.6, source_id="src/a")
    _ = jsonl.write_text(row.model_dump_json() + "\n\n")

    manifest_path = tmp_path / "eval_dataset_manifest.json"
    manifest = EvalSetManifest(
        source_search_tag="blank-line",
        hardware=_HARDWARE,
        harvested_per_bin={SpeedupBin.MINOR_SPEEDUP: 1},
        generated_per_bin={},
        attempts_per_bin={},
        generated_at=datetime(2026, 4, 29, tzinfo=timezone.utc),
    )
    _ = manifest_path.write_text(manifest.model_dump_json())

    loaded = read_eval_dataset(tmp_path)
    assert len(loaded.comparisons) == 1
