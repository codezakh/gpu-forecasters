from __future__ import annotations

from collections import defaultdict
from typing import TypedDict, cast

from pydantic import BaseModel

from gpu_forecasters.landscape_map.v1.domain import KernelImplementation, KernelTaskInfo

# Type alias for a task key: (level_id, task_id, op_name)
TaskKey = tuple[int, int, str]


class ArchiveRow(TypedDict):
    Level_ID: int | str
    Task_ID: int | str
    Op_Name: str
    PyTorch_Code_Functional: str | None
    PyTorch_Native_Runtime: float | int | None
    Correct: bool
    CUDA_Runtime: float | int | None
    Kernel_Name: str
    CUDA_Code: str


class KernelSample(BaseModel, frozen=True):
    """A single kernel pair sample from the Sakana archive."""

    task: KernelTaskInfo
    reference: KernelImplementation
    candidate: KernelImplementation
    speedup: float  # reference_runtime_ms / candidate_runtime_ms


def load_archive_rows(levels: list[int]) -> dict[TaskKey, list[ArchiveRow]]:
    """Load rows from the Sakana AI-CUDA-Engineer-Archive dataset.

    Groups rows by (level_id, task_id, op_name) task key.
    """
    import datasets

    split_names = [f"level_{lvl}" for lvl in levels]
    task_rows: dict[TaskKey, list[ArchiveRow]] = defaultdict(list)

    for split in split_names:
        ds = datasets.load_dataset(
            "SakanaAI/AI-CUDA-Engineer-Archive",
            split=split,
        )
        for row_untyped in ds:
            row = cast(ArchiveRow, row_untyped)
            key: TaskKey = (
                int(row["Level_ID"]),
                int(row["Task_ID"]),
                str(row["Op_Name"]),
            )
            task_rows[key].append(row)

    return dict(task_rows)
