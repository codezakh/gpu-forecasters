"""Core domain types and protocols for the v2 landscape map surrogate.

Differences from v1:
  - the surrogate's per-bin uncertainty is a numerical probability
    distribution (a true simplex over bins 1..8), not a Likert scale;
  - the estimator output carries the model's raw, pre-renormalization
    probability sum so calibration scoring can surface how far the
    distribution was from a true simplex before renormalization.
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import Mapping, Protocol

from pydantic import BaseModel, Field, model_validator


# ---------------------------------------------------------------------------
# Domain primitives
# ---------------------------------------------------------------------------


class KernelTaskInfo(BaseModel, frozen=True):
    op_name: str
    level_id: int
    task_id: int


class KernelImplementation(BaseModel, frozen=True):
    kernel_name: str
    code: str
    runtime_ms: float | None


class HardwareContext(BaseModel, frozen=True):
    """Hardware parameters that condition the runtime estimate.

    Populated from torch.cuda device properties at eval-set build time.
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


# ---------------------------------------------------------------------------
# Speedup bins
# ---------------------------------------------------------------------------


class SpeedupBin(IntEnum):
    """Discretized speedup bins using half-octave (k=2) resolution.

    Bin index is computed as ``floor(2 * log2(S)) + 4``, clamped to
    ``[1, 8]``. Bin 0 is reserved for failures (compilation error or
    incorrect output) — the surrogate's predicted distribution does
    *not* include bin 0 as an outcome.
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
        """Map a continuous speedup ratio to a discrete bin in [1, 8].

        Does NOT return FAILURE — that requires external correctness
        information.
        """
        if speedup <= 0:
            raise ValueError(f"Speedup must be positive, got {speedup}")
        raw_index = math.floor(2 * math.log2(speedup)) + 4
        clamped = max(1, min(8, raw_index))
        return cls(clamped)

    @property
    def label(self) -> str:
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


# Bins the surrogate's distribution covers (success bins only — bin 0
# is excluded by construction).
SUCCESS_BINS: tuple[SpeedupBin, ...] = (
    SpeedupBin.SEVERE_SLOWDOWN,
    SpeedupBin.SIGNIFICANT_SLOWDOWN,
    SpeedupBin.MODERATE_SLOWDOWN,
    SpeedupBin.MINOR_SLOWDOWN,
    SpeedupBin.MINOR_SPEEDUP,
    SpeedupBin.SIGNIFICANT_SPEEDUP,
    SpeedupBin.HIGH_SPEEDUP,
    SpeedupBin.EXTREME_SPEEDUP,
)


# ---------------------------------------------------------------------------
# Estimate output
# ---------------------------------------------------------------------------


class KernelRuntimeEstimate(BaseModel, frozen=True):
    """Structured output of a v2 SpeedupEstimator.

    ``bin_probabilities`` covers exactly :data:`SUCCESS_BINS` and is
    guaranteed to sum to 1 within ``1e-6`` (renormalized at parse
    time). ``raw_probability_sum`` is what the model emitted *before*
    renormalization, useful as a calibration-health signal.
    """

    predicted_bin: SpeedupBin
    bin_probabilities: dict[SpeedupBin, float]
    reasoning: str
    raw_probability_sum: float = Field(
        ge=0.0,
        description=(
            "Sum of the model's raw probabilities before renormalization. "
            "A value far from 1 indicates poor calibration even when the "
            "renormalized distribution is well-formed."
        ),
    )

    @model_validator(mode="after")
    def _check_invariants(self) -> KernelRuntimeEstimate:
        if self.predicted_bin == SpeedupBin.FAILURE:
            raise ValueError(
                "predicted_bin must be one of SUCCESS_BINS (1..8), got FAILURE"
            )
        # Keys must be exactly SUCCESS_BINS — no missing, no extra.
        if set(self.bin_probabilities.keys()) != set(SUCCESS_BINS):
            missing = sorted(set(SUCCESS_BINS) - set(self.bin_probabilities.keys()))
            extra = sorted(set(self.bin_probabilities.keys()) - set(SUCCESS_BINS))
            raise ValueError(
                f"bin_probabilities must cover exactly bins 1..8 "
                f"(missing={missing}, extra={extra})"
            )
        # All probabilities must be non-negative (they may be small but
        # exactly-zero positive after renormalization, so >=0 not >0).
        for b, p in self.bin_probabilities.items():
            if p < 0.0:
                raise ValueError(
                    f"bin_probabilities[{b}] = {p} is negative"
                )
        # Renormalized sum is 1 within float tolerance.
        total = sum(self.bin_probabilities.values())
        if not math.isclose(total, 1.0, abs_tol=1e-6):
            raise ValueError(
                f"bin_probabilities must sum to 1 within 1e-6, got {total!r}"
            )
        return self

    def probability_of(self, bin_: SpeedupBin) -> float:
        """Return the probability mass on ``bin_``.

        ``SpeedupBin.FAILURE`` returns 0 by construction (the
        distribution is over success bins only).
        """
        if bin_ == SpeedupBin.FAILURE:
            return 0.0
        return self.bin_probabilities[bin_]

    def bin_probability_list(self) -> list[float]:
        """Probabilities ordered by bin index (1..8)."""
        return [self.bin_probabilities[b] for b in SUCCESS_BINS]


def renormalize(
    raw_probabilities: Mapping[SpeedupBin, float],
) -> tuple[dict[SpeedupBin, float], float]:
    """Renormalize a non-negative measure over success bins to a simplex.

    Returns the renormalized distribution and the original sum. The
    caller is responsible for ensuring keys cover :data:`SUCCESS_BINS`
    and that values are non-negative; this function trusts both.
    """
    total = float(sum(raw_probabilities.values()))
    if total <= 0:
        raise ValueError(
            f"Cannot renormalize: probabilities sum to {total} (expected > 0)"
        )
    return ({b: float(raw_probabilities[b]) / total for b in SUCCESS_BINS}, total)


# ---------------------------------------------------------------------------
# Component protocols
# ---------------------------------------------------------------------------


class SpeedupEstimator(Protocol):
    """Protocol for synchronously estimating relative kernel speedup."""

    def estimate(
        self, query: KernelRuntimeQuery
    ) -> tuple[KernelRuntimeEstimate, LlmCallUsage | None]: ...


class AsyncSpeedupEstimator(Protocol):
    """Protocol for asynchronously estimating relative kernel speedup."""

    async def aestimate(
        self, query: KernelRuntimeQuery
    ) -> tuple[KernelRuntimeEstimate, LlmCallUsage | None]: ...
