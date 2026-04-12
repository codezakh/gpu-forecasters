from __future__ import annotations

import math
from enum import Enum, IntEnum
from typing import Protocol

from pydantic import BaseModel


class KernelTaskInfo(BaseModel, frozen=True):
    op_name: str
    level_id: int
    task_id: int


class KernelImplementation(BaseModel, frozen=True):
    kernel_name: str  # "pytorch_functional" for PyTorch ref
    code: str
    runtime_ms: float | None  # None only when strategy=ANY_FROM_TASK for PyTorch ref


class SpeedupBin(IntEnum):
    """Discretized speedup bins using half-octave (k=2) resolution.

    Bin index is computed as: floor(2 * log2(S)) + 4, clamped to [1, 8].
    Bin 0 is reserved for failures (compilation error or incorrect output).
    """

    FAILURE = 0
    SEVERE_SLOWDOWN = 1  # S <= 0.25
    SIGNIFICANT_SLOWDOWN = 2  # 0.25 < S <= 0.5
    MODERATE_SLOWDOWN = 3  # 0.5 < S <= 0.707
    MINOR_SLOWDOWN = 4  # 0.707 < S <= 1.0
    MINOR_SPEEDUP = 5  # 1.0 < S <= 1.414
    SIGNIFICANT_SPEEDUP = 6  # 1.414 < S <= 2.0
    HIGH_SPEEDUP = 7  # 2.0 < S <= 4.0
    EXTREME_SPEEDUP = 8  # S > 4.0

    @classmethod
    def from_speedup(cls, speedup: float) -> SpeedupBin:
        """Map a continuous speedup ratio to a discrete bin.

        Uses the formula: i = floor(2 * log2(S)) + 4, clamped to [1, 8].
        Does NOT return FAILURE — that requires external correctness info.
        """
        if speedup <= 0:
            raise ValueError(f"Speedup must be positive, got {speedup}")
        raw_index = math.floor(2 * math.log2(speedup)) + 4
        clamped = max(1, min(8, raw_index))
        return cls(clamped)

    @property
    def label(self) -> str:
        """Human-readable label for this bin."""
        return _BIN_LABELS[self]


_BIN_LABELS: dict[SpeedupBin, str] = {
    SpeedupBin.FAILURE: "failure",
    SpeedupBin.SEVERE_SLOWDOWN: "severe slowdown (≤0.25x)",
    SpeedupBin.SIGNIFICANT_SLOWDOWN: "significant slowdown (0.25x–0.5x)",
    SpeedupBin.MODERATE_SLOWDOWN: "moderate slowdown (0.5x–0.71x)",
    SpeedupBin.MINOR_SLOWDOWN: "minor slowdown (0.71x–1.0x)",
    SpeedupBin.MINOR_SPEEDUP: "minor speedup (1.0x–1.41x)",
    SpeedupBin.SIGNIFICANT_SPEEDUP: "significant speedup (1.41x–2.0x)",
    SpeedupBin.HIGH_SPEEDUP: "high speedup (2.0x–4.0x)",
    SpeedupBin.EXTREME_SPEEDUP: "extreme speedup (>4.0x)",
}


class LikertConfidence(str, Enum):
    """5-point Likert scale for verbalized confidence."""

    VERY_LOW = "very_low"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class HardwareContext(BaseModel, frozen=True):
    """Hardware parameters that condition the runtime estimate.

    Populated from torch.cuda device properties.
    """

    device_name: str
    compute_capability: tuple[int, int]
    total_global_memory_gb: float
    multiprocessor_count: int
    max_threads_per_multiprocessor: int
    clock_rate_ghz: float
    memory_clock_rate_ghz: float
    memory_bus_width_bits: int


class KernelRuntimeQuery(BaseModel, frozen=True):
    """Input to a SpeedupEstimator: the kernel pair plus conditioning info."""

    task: KernelTaskInfo
    reference: KernelImplementation
    candidate: KernelImplementation
    hardware: HardwareContext | None = None


class LlmCallUsage(BaseModel, frozen=True):
    """Token usage for a single LLM API call."""

    input_tokens: int
    output_tokens: int


class KernelRuntimeEstimate(BaseModel, frozen=True):
    """Structured output of a SpeedupEstimator."""

    predicted_bin: SpeedupBin
    bin_confidences: dict[SpeedupBin, LikertConfidence]
    reasoning: str


class SpeedupEstimator(Protocol):
    """Protocol for estimating relative kernel speedup."""

    def estimate(
        self, query: KernelRuntimeQuery
    ) -> tuple[KernelRuntimeEstimate, LlmCallUsage | None]: ...


class AsyncSpeedupEstimator(Protocol):
    """Protocol for asynchronously estimating relative kernel speedup."""

    async def aestimate(
        self, query: KernelRuntimeQuery
    ) -> tuple[KernelRuntimeEstimate, LlmCallUsage | None]: ...
