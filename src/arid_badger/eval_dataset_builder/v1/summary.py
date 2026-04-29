"""Run summary computation: fold an event log into per-bin distributions
and extract the in-target ``KernelRuntimeComparison`` rows the search
produced. Pack-generic.
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Mapping

from arid_badger.gpu_mode_kernel.core import (
    CaseSpeedupT,
    GpuModeKernelObservation,
    InfrastructureFailureFeedback,
    SuccessFeedback,
)
from arid_badger.landscape_map.v1.domain import HardwareContext, SpeedupBin
from arid_badger.max_reward_puct.v2.events import (
    EvaluationCompleted,
    EvaluationRequested,
    SearchEvent,
)
from arid_badger.max_reward_puct.v2.state import replay

from .domain import KernelRuntimeComparison, RunSummary


_FAILURE_KEY = "FAILURE"
_INFRASTRUCTURE_FAILURE_KEY = "INFRASTRUCTURE_FAILURE"


def _classify_observation(
    observation: GpuModeKernelObservation[CaseSpeedupT],
) -> str:
    feedback = observation.feedback
    if isinstance(feedback, SuccessFeedback):
        return SpeedupBin.from_speedup(feedback.aggregated_speedup).name
    if isinstance(feedback, InfrastructureFailureFeedback):
        return _INFRASTRUCTURE_FAILURE_KEY
    return _FAILURE_KEY


def compute_run_summary_from_event_log(
    events: list[SearchEvent[GpuModeKernelObservation[CaseSpeedupT]]],
    *,
    target_bin: SpeedupBin,
    target_band_lo: float,
    target_band_hi: float,
    target_midpoint_speedup: float,
    seed_source_id: str | None,
    seed_speedup_at_harvest: float | None,
    model_slug: str,
    search_config: Mapping[str, object],
    k_per_parent: int,
    archive_capacity: int,
    wall_clock_seconds: float,
    observation_type: type[GpuModeKernelObservation[CaseSpeedupT]],
) -> RunSummary:
    candidate_bin_counter: Counter[str] = Counter()
    seed_speedup_after_bootstrap: float | None = None

    for event in events:
        if isinstance(event, EvaluationCompleted):
            candidate_bin_counter[
                _classify_observation(event.evaluation.observation)
            ] += 1

    state = replay(
        events,
        k_per_parent=k_per_parent,
        archive_capacity=archive_capacity,
        observation_type=observation_type,
    )

    if state.archive:
        # The seed (root) is the only seed in this run; its post-bootstrap
        # speedup is whatever the wrapped evaluation's observation carries.
        for node in state.archive:
            if node.is_seed:
                feedback = node.evaluation.observation.feedback
                if isinstance(feedback, SuccessFeedback):
                    seed_speedup_after_bootstrap = feedback.aggregated_speedup
                break

    archive_bin_counter: Counter[str] = Counter()
    for node in state.archive:
        archive_bin_counter[_classify_observation(node.evaluation.observation)] += 1

    target_bin_name = target_bin.name
    return RunSummary(
        target_bin=target_bin_name,
        target_band_lo=target_band_lo,
        target_band_hi=target_band_hi,
        target_midpoint_speedup=target_midpoint_speedup,
        seed_source_id=seed_source_id,
        seed_speedup_at_harvest=seed_speedup_at_harvest,
        seed_speedup_after_bootstrap_eval=seed_speedup_after_bootstrap,
        model_slug=model_slug,
        search_config=dict(search_config),
        total_candidates_evaluated=sum(candidate_bin_counter.values()),
        per_bin_count_all_candidates=dict(candidate_bin_counter),
        per_bin_count_archive_at_end=dict(archive_bin_counter),
        in_target_bin_count_all_candidates=candidate_bin_counter.get(target_bin_name, 0),
        in_target_bin_count_archive_at_end=archive_bin_counter.get(target_bin_name, 0),
        wall_clock_seconds=wall_clock_seconds,
    )


def extract_in_target_kernels_from_event_log(
    events: list[SearchEvent[GpuModeKernelObservation[CaseSpeedupT]]],
    *,
    target_bin: SpeedupBin,
    reference_code: str,
    hardware: HardwareContext,
    source_id_prefix: str,
) -> list[KernelRuntimeComparison]:
    """Walk the event log and produce one ``KernelRuntimeComparison`` per
    in-target success-arm ``EvaluationCompleted``.

    Excludes the bootstrap eval (the seed) — the seed already came from
    the harvest pool, so including it would double-count. Each output
    row's ``source_id`` is ``"{source_id_prefix}/{sha12}"`` where sha12
    is the first 12 hex chars of the candidate code's SHA-256.
    """
    code_by_request: dict[str, str] = {}
    out: list[KernelRuntimeComparison] = []
    for event in events:
        if isinstance(event, EvaluationRequested):
            code_by_request[event.request_id] = event.code
        elif isinstance(event, EvaluationCompleted):
            feedback = event.evaluation.observation.feedback
            if not isinstance(feedback, SuccessFeedback):
                continue
            speedup = feedback.aggregated_speedup
            measured_bin = SpeedupBin.from_speedup(speedup)
            if measured_bin != target_bin:
                continue
            code = code_by_request.get(event.request_id)
            if code is None:
                # Bootstrap eval — has no EvaluationRequested; skip.
                continue
            sha12 = hashlib.sha256(code.encode("utf-8")).hexdigest()[:12]
            out.append(
                KernelRuntimeComparison(
                    reference_code=reference_code,
                    candidate_code=code,
                    hardware=hardware,
                    aggregated_speedup=speedup,
                    true_bin=measured_bin,
                    source_id=f"{source_id_prefix}/{sha12}",
                )
            )
    return out
