"""``V2CheckpointSource`` tests.

Round-trip a synthesized v2 checkpoint, then verify the source emits
one ``KernelRuntimeComparison`` per success node, drops compile-failed
nodes, and pins the metadata claims (reference, hardware, source tag)
correctly.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import TypeAdapter
from ulid import ULID

from arid_badger.eval_dataset_builder.v1 import V2CheckpointSource
from arid_badger.gpu_mode_kernel.core import (
    CompileFailedFeedback,
    GpuModeKernelObservation,
    SuccessFeedback,
)
from arid_badger.gpu_mode_kernel.packs.trimul import TriMulCaseSpeedup
from arid_badger.hill_climbing.domain import Evaluation, Node
from arid_badger.landscape_map.v1.domain import HardwareContext, SpeedupBin
from arid_badger.max_reward_puct.checkpoint import PuctCheckpoint


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


def _success_node(*, code: str, speedup: float) -> Node[GpuModeKernelObservation[TriMulCaseSpeedup]]:
    feedback = SuccessFeedback[TriMulCaseSpeedup](
        aggregated_speedup=speedup,
        aggregation_method="geomean",
        per_case_speedups=[],
    )
    return Node[GpuModeKernelObservation[TriMulCaseSpeedup]](
        program_code=code,
        ancestors=[],
        evaluation=Evaluation[GpuModeKernelObservation[TriMulCaseSpeedup]](
            observation=GpuModeKernelObservation[TriMulCaseSpeedup](feedback=feedback),
            reward=speedup,
        ),
        ulid=ULID(),
    )


def _failure_node(*, code: str) -> Node[GpuModeKernelObservation[TriMulCaseSpeedup]]:
    feedback = CompileFailedFeedback(compilation_error="boom")
    return Node[GpuModeKernelObservation[TriMulCaseSpeedup]](
        program_code=code,
        ancestors=[],
        evaluation=Evaluation[GpuModeKernelObservation[TriMulCaseSpeedup]](
            observation=GpuModeKernelObservation[TriMulCaseSpeedup](feedback=feedback),
            reward=None,
        ),
        ulid=ULID(),
    )


def _write_checkpoint(
    path: Path,
    nodes: list[Node[GpuModeKernelObservation[TriMulCaseSpeedup]]],
) -> None:
    checkpoint = PuctCheckpoint[GpuModeKernelObservation[TriMulCaseSpeedup]](
        archive=nodes,
        seed_ids=set(),
        visit_counts={},
        best_child_rewards={},
        global_expansion_count=0,
        current_step=0,
    )
    adapter: TypeAdapter[PuctCheckpoint[GpuModeKernelObservation[TriMulCaseSpeedup]]] = (
        TypeAdapter(PuctCheckpoint[GpuModeKernelObservation[TriMulCaseSpeedup]])
    )
    _ = path.write_bytes(adapter.dump_json(checkpoint))


def test_yields_one_comparison_per_success_node(tmp_path: Path) -> None:
    success_a = _success_node(code="A", speedup=2.5)
    success_b = _success_node(code="B", speedup=0.4)
    failure = _failure_node(code="C")
    path = tmp_path / "checkpoint.json"
    _write_checkpoint(path, [success_a, failure, success_b])

    source = V2CheckpointSource[TriMulCaseSpeedup](
        checkpoint_path=path,
        case_speedup_type=TriMulCaseSpeedup,
        reference_code="REF",
        hardware=_HARDWARE,
        source_search_tag="eXXXX",
    )

    rows = list(source())

    assert len(rows) == 2
    by_code = {row.candidate_code: row for row in rows}
    assert set(by_code) == {"A", "B"}
    assert by_code["A"].aggregated_speedup == 2.5
    assert by_code["A"].true_bin == SpeedupBin.from_speedup(2.5)
    assert by_code["A"].reference_code == "REF"
    assert by_code["A"].hardware == _HARDWARE
    assert by_code["A"].source_id == f"eXXXX/{success_a.ulid}"
    assert by_code["B"].true_bin == SpeedupBin.from_speedup(0.4)


def test_empty_archive_yields_nothing(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint.json"
    _write_checkpoint(path, [])
    source = V2CheckpointSource[TriMulCaseSpeedup](
        checkpoint_path=path,
        case_speedup_type=TriMulCaseSpeedup,
        reference_code="REF",
        hardware=_HARDWARE,
        source_search_tag="eXXXX",
    )
    assert list(source()) == []
