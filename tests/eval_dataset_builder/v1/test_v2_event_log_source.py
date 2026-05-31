"""``V2EventLogSource`` tests.

Drive a real ``FileEventLog`` with a hand-built sequence of v2
``SearchEvent``s, then verify the source replays the log into the
expected archive: one ``KernelRuntimeComparison`` per success node,
compile-failed nodes dropped, metadata pinned correctly.
"""

from __future__ import annotations

from pathlib import Path

from ulid import ULID

from gpu_forecasters.eval_dataset_builder.v1 import V2EventLogSource
from gpu_forecasters.gpu_mode_kernel.core import (
    CompileFailedFeedback,
    GpuModeKernelObservation,
    SuccessFeedback,
)
from gpu_forecasters.gpu_mode_kernel.packs.trimul import TriMulCaseSpeedup
from gpu_forecasters.hill_climbing.domain import Evaluation, Node
from gpu_forecasters.landscape_map.v1.domain import HardwareContext, SpeedupBin
from gpu_forecasters.max_reward_puct.v2.event_log import FileEventLog
from gpu_forecasters.max_reward_puct.v2.events import (
    EvaluationCompleted,
    EvaluationRequested,
    SearchInitialized,
    StepCompleted,
    StepStarted,
)


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

ObservationT = GpuModeKernelObservation[TriMulCaseSpeedup]


def _success_evaluation(*, speedup: float) -> Evaluation[ObservationT]:
    feedback = SuccessFeedback[TriMulCaseSpeedup](
        aggregated_speedup=speedup,
        aggregation_method="geomean",
        per_case_speedups=[],
    )
    return Evaluation[ObservationT](
        observation=ObservationT(feedback=feedback),
        reward=speedup,
    )


def _failure_evaluation() -> Evaluation[ObservationT]:
    return Evaluation[ObservationT](
        observation=ObservationT(feedback=CompileFailedFeedback(compilation_error="boom")),
        reward=None,
    )


def _seed_node(*, code: str, speedup: float) -> Node[ObservationT]:
    return Node[ObservationT](
        program_code=code,
        ancestors=[],
        evaluation=_success_evaluation(speedup=speedup),
        is_seed=True,
        ulid=ULID(),
    )


def _write_log(path: Path) -> tuple[Node[ObservationT], ULID, ULID, ULID]:
    """Build a minimal but valid v2 event sequence:

    seed (success) → step 1 expands seed with two children, one
    success and one compile failure → step done.
    """
    log: FileEventLog[ObservationT] = FileEventLog(path, observation_type=ObservationT)
    seed = _seed_node(code="SEED", speedup=1.0)
    log.append(SearchInitialized[ObservationT](root=seed))

    log.append(StepStarted(step=1, parent_ulids=[seed.ulid]))
    success_ulid = ULID()
    failure_ulid = ULID()
    success_req = "req-success"
    failure_req = "req-failure"
    log.append(
        EvaluationRequested(
            request_id=success_req,
            child_ulid=success_ulid,
            parent_ulid=seed.ulid,
            code="A",
        )
    )
    log.append(
        EvaluationRequested(
            request_id=failure_req,
            child_ulid=failure_ulid,
            parent_ulid=seed.ulid,
            code="C",
        )
    )
    log.append(
        EvaluationCompleted[ObservationT](
            request_id=success_req,
            evaluation=_success_evaluation(speedup=2.5),
        )
    )
    log.append(
        EvaluationCompleted[ObservationT](
            request_id=failure_req,
            evaluation=_failure_evaluation(),
        )
    )
    log.append(StepCompleted(step=1))
    return seed, seed.ulid, success_ulid, failure_ulid


def test_yields_one_comparison_per_archive_success(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    seed, seed_ulid, success_ulid, _failure_ulid = _write_log(path)

    source: V2EventLogSource[TriMulCaseSpeedup] = V2EventLogSource(
        events_path=path,
        case_speedup_type=TriMulCaseSpeedup,
        k_per_parent=2,
        archive_capacity=32,
        reference_code="REF",
        hardware=_HARDWARE,
        source_search_tag="eXXXX",
    )

    rows = list(source())

    by_code = {row.candidate_code: row for row in rows}
    # SEED (1.0x) and A (2.5x) survive; failure C is dropped at fold time.
    assert set(by_code) == {"SEED", "A"}
    assert by_code["A"].aggregated_speedup == 2.5
    assert by_code["A"].true_bin == SpeedupBin.from_speedup(2.5)
    assert by_code["A"].reference_code == "REF"
    assert by_code["A"].hardware == _HARDWARE
    assert by_code["A"].source_id == f"eXXXX/{success_ulid}"
    assert by_code["SEED"].source_id == f"eXXXX/{seed_ulid}"
    assert seed is not None  # silence unused


def test_empty_log_yields_nothing(tmp_path: Path) -> None:
    path = tmp_path / "events.jsonl"
    path.touch()
    source: V2EventLogSource[TriMulCaseSpeedup] = V2EventLogSource(
        events_path=path,
        case_speedup_type=TriMulCaseSpeedup,
        k_per_parent=2,
        archive_capacity=32,
        reference_code="REF",
        hardware=_HARDWARE,
        source_search_tag="eXXXX",
    )
    assert list(source()) == []
