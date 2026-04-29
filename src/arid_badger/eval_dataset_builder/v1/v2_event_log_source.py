"""``HarvestedKernelSource`` for v2 max-reward-PUCT event logs.

v2 PUCT's authoritative durable artifact is ``events.jsonl`` (see
``arid_badger.max_reward_puct.v2.event_log.FileEventLog``). v2 does
not write a ``PuctCheckpoint`` JSON; the archive is reconstructed by
folding events through ``arid_badger.max_reward_puct.v2.state.replay``.

This adapter wraps that fold so every gpu-mode pack can hand the
eval-set builder a v2 events log directly. It is the v2-native
counterpart to the legacy v1 per-experiment checkpoint adapters
(e.g. ``E0034CheckpointSource``).
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Generic, final

from arid_badger.gpu_mode_kernel.core import CaseSpeedupT, GpuModeKernelObservation
from arid_badger.landscape_map.v1.domain import HardwareContext, SpeedupBin
from arid_badger.max_reward_puct.v2.event_log import FileEventLog
from arid_badger.max_reward_puct.v2.state import replay

from .domain import KernelRuntimeComparison


@final
class V2EventLogSource(Generic[CaseSpeedupT]):
    """Reads a v2 PUCT ``events.jsonl`` and yields archive successes as
    ``KernelRuntimeComparison``s.

    Replay parameters (``k_per_parent``, ``archive_capacity``) must
    match the values used during the source search — they parameterize
    the same reducer that produced the archive at search time.

    Successes-only by construction: the v2 archive only retains nodes
    whose evaluation succeeded; compile/runtime failures are dropped at
    fold time.
    """

    def __init__(
        self,
        *,
        events_path: Path,
        case_speedup_type: type[CaseSpeedupT],
        k_per_parent: int,
        archive_capacity: int,
        reference_code: str,
        hardware: HardwareContext,
        source_search_tag: str,
    ) -> None:
        self._events_path = events_path
        self._observation_type: type[GpuModeKernelObservation[CaseSpeedupT]] = (
            GpuModeKernelObservation[case_speedup_type]
        )
        self._k_per_parent = k_per_parent
        self._archive_capacity = archive_capacity
        self._reference_code = reference_code
        self._hardware = hardware
        self._source_search_tag = source_search_tag

    def __call__(self) -> Iterable[KernelRuntimeComparison]:
        event_log: FileEventLog[GpuModeKernelObservation[CaseSpeedupT]] = FileEventLog(
            self._events_path, observation_type=self._observation_type
        )
        state = replay(
            event_log.read_all(),
            k_per_parent=self._k_per_parent,
            archive_capacity=self._archive_capacity,
            observation_type=self._observation_type,
        )
        for node in state.archive:
            feedback = node.evaluation.observation.feedback
            if feedback.kind != "success":
                continue
            speedup = feedback.aggregated_speedup
            yield KernelRuntimeComparison(
                reference_code=self._reference_code,
                candidate_code=node.program_code,
                hardware=self._hardware,
                aggregated_speedup=speedup,
                true_bin=SpeedupBin.from_speedup(speedup),
                source_id=f"{self._source_search_tag}/{node.ulid}",
            )
