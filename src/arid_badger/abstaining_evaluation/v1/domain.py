"""Domain types and protocols for the abstaining-evaluation library.

The seam this module defines:

* ``PredictOrAbstain`` is the per-row decision an abstainer produces.
* ``AbstainPolicy`` is the protocol an abstainer satisfies — anything
  that can map an estimate (or its absence) to a decision.
* ``ConfidenceScore`` and ``RiskFunction`` are the two seams the rest of
  the library reaches for. ``ConfidenceScore`` shapes the threshold
  abstainer; ``RiskFunction`` shapes the metrics.
"""

from __future__ import annotations

from typing import Annotated, Literal, Protocol, Sequence, Union

from pydantic import BaseModel, ConfigDict, Field

from arid_badger.eval_dataset_builder.v1 import KernelRuntimeComparison
from arid_badger.landscape_map.v2 import KernelRuntimeEstimate


class Predict(BaseModel):
    """The abstainer chose to forecast for this row.

    Carries the estimate it forecasts plus the row's ground-truth
    fields (denormalized so downstream metric code does not need a
    parallel index into the comparisons list).
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["predict"] = "predict"
    estimate: KernelRuntimeEstimate


class AbstainDecision(BaseModel):
    """The abstainer chose to defer this row to the real evaluator.

    ``reason`` is optional and currently used only by the native
    abstainer (which can carry an LLM rationale) — for the threshold
    abstainer it is left as None.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["abstain"] = "abstain"
    reason: str | None = None


PredictOrAbstain = Annotated[
    Union[Predict, AbstainDecision],
    Field(discriminator="kind"),
]


class RiskCoveragePoint(BaseModel):
    """One (coverage, risk) point on a sweepable curve.

    ``threshold`` is the threshold value that produced this point in a
    threshold sweep, surfaced for diagnostics. It is ``None`` for
    points that were not produced by a sweep (e.g. the realized point
    of a native abstainer).
    """

    model_config = ConfigDict(frozen=True)

    coverage: float
    risk: float
    n_predicted: int
    n_abstained: int
    threshold: float | None = None


class SelectiveMetricRow(BaseModel):
    """One scalar rollup row for a (label, risk_function) crosstab.

    ``aurc`` is the area under the risk-coverage curve;
    ``selective_at_50`` is the linearly-interpolated risk at 50%
    coverage. Both are unitless from the pointwise risks (binary,
    speedup-distance) and in speedup units for regret.
    """

    model_config = ConfigDict(frozen=True)

    label: str
    risk_name: str
    aurc: float
    selective_at_50: float
    n_total: int


class ConfidenceScore(Protocol):
    """A real-valued confidence assigned to one estimate.

    Larger = more confident. The threshold abstainer abstains when
    ``score(estimate) < threshold``, so any score function plugs in
    transparently.
    """

    def __call__(self, estimate: KernelRuntimeEstimate) -> float: ...


class AbstainPolicy(Protocol):
    """Maps an estimate (or its absence) to a predict-or-abstain decision."""

    def __call__(
        self, estimate: KernelRuntimeEstimate | None
    ) -> PredictOrAbstain: ...


class RiskFunction(Protocol):
    """Aggregate risk over a row-aligned (decisions, comparisons) pair.

    Pointwise risks (binary mismatch, speedup distance) are mean-of-
    pointwise-loss over the predicted subset. Set-level risks (regret)
    can use both the predicted and abstained subsets.
    """

    @property
    def name(self) -> str: ...

    def __call__(
        self,
        decisions: Sequence[PredictOrAbstain],
        comparisons: Sequence[KernelRuntimeComparison],
    ) -> float: ...
