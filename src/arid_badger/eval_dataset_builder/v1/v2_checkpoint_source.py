"""``HarvestedKernelSource`` for v2 max-reward-PUCT checkpoints.

v2 PUCT writes ``PuctCheckpoint[GpuModeKernelObservation[CaseSpeedupT]]``
for every ``KernelPack`` — one shape, parameterized only by the pack's
per-case speedup type. This adapter is the matching reader: every
gpu-mode pack can use it by passing its ``case_speedup_type`` at
construction.

Legacy checkpoint formats (e.g. e0034's ``TriMulObservation`` shape)
keep their own per-experiment adapters; the library does not replace
those.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Generic, final

from pydantic import TypeAdapter

from arid_badger.gpu_mode_kernel.core import CaseSpeedupT, GpuModeKernelObservation
from arid_badger.landscape_map.v1.domain import HardwareContext, SpeedupBin
from arid_badger.max_reward_puct.checkpoint import PuctCheckpoint

from .domain import KernelRuntimeComparison


@final
class V2CheckpointSource(Generic[CaseSpeedupT]):
    """Reads a v2 max-reward-PUCT checkpoint and yields successes-only as
    ``KernelRuntimeComparison``s.

    The v2 archive only retains successes — failures were dropped at
    write time — so this adapter is successes-only by construction,
    matching the eval set's invariant.
    """

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        case_speedup_type: type[CaseSpeedupT],
        reference_code: str,
        hardware: HardwareContext,
        source_search_tag: str,
    ) -> None:
        # TypeAdapter on the parametrized generic — required so the
        # nested ``Evaluation[GpuModeKernelObservation[CaseSpeedupT]]``
        # deserializes against the pack's concrete per-case speedup
        # schema rather than the TypeVar default. Same rationale as
        # ``FilePuctCheckpointProvider``.
        self._adapter: TypeAdapter[
            PuctCheckpoint[GpuModeKernelObservation[CaseSpeedupT]]
        ] = TypeAdapter(
            PuctCheckpoint[GpuModeKernelObservation[case_speedup_type]]
        )
        self._checkpoint_path = checkpoint_path
        self._reference_code = reference_code
        self._hardware = hardware
        self._source_search_tag = source_search_tag

    def __call__(self) -> Iterable[KernelRuntimeComparison]:
        checkpoint = self._adapter.validate_json(self._checkpoint_path.read_text())
        for node in checkpoint.archive:
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
