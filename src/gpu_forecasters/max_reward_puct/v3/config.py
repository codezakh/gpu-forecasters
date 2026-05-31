"""Configuration objects for v3 max-reward PUCT.

A caller constructs one ``SearchConfig``, hands it to the driver, and
doesn't worry about individual knobs at the call site. Group knobs that
belong together into sub-objects once there's enough of them to warrant
it.

The ranking rule that powers surrogate-driven selection is part of the
config rather than the driver constructor: it is a *decision policy*,
not infrastructure. A search re-run from the same log+config produces
the same selection scores, by construction.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from pydantic import BaseModel, ConfigDict

from gpu_forecasters.hill_climbing.domain import Node, ObservationT
from gpu_forecasters.landscape_map.v2 import KernelRuntimeEstimate, SUCCESS_BINS


@runtime_checkable
class RankingRule(Protocol):
    """Pure ranking rule consumed by ``compute_pending_actions`` at the
    selection barrier.

    Takes one candidate's forecast plus a snapshot of archive state at
    the moment of selection, returns a scalar score. Higher is better.

    Implementations must be deterministic functions of their inputs;
    no clock, no randomness, no external state. Spec § principles.
    """

    def score(
        self,
        *,
        forecast: KernelRuntimeEstimate,
        archive: Sequence[Node[ObservationT]],
    ) -> float: ...


class ExpectedBinIndexRule:
    """Score = sum_b b * P(b). Higher b means higher predicted speedup.

    Independent of archive state — the simplest rule that uses the
    full distribution rather than just the argmax bin. Useful as a
    baseline and for tests where empirical-midpoint logic is overkill.
    """

    def score(
        self,
        *,
        forecast: KernelRuntimeEstimate,
        archive: Sequence[Node[ObservationT]],
    ) -> float:
        del archive
        return sum(int(b) * p for b, p in forecast.bin_probabilities.items())


class PredictedBinRule:
    """Score = predicted_bin (1..8). Argmax of the distribution.

    Even simpler than the expected-bin rule; useful when the forecast
    is concentrated and we want the rule to be discrete.
    """

    def score(
        self,
        *,
        forecast: KernelRuntimeEstimate,
        archive: Sequence[Node[ObservationT]],
    ) -> float:
        del archive
        return float(int(forecast.predicted_bin))


# Ranking-rule registry — exposes the bin set so callers can sanity-
# check that a rule's outputs cover what they expect.
__all__ = [
    "ExpectedBinIndexRule",
    "PredictedBinRule",
    "RankingRule",
    "SUCCESS_BINS",
    "SearchConfig",
]


class SearchConfig(BaseModel):
    """All parameters that shape the search itself. Providers, the
    surrogate, and the event log are injected separately — they are
    infrastructure, not config.

    ``samples_per_parent`` is the number of mutations dispatched per
    parent per step. ``k_per_parent`` is the per-parent GPU-evaluation
    budget after surrogate filtering: at most ``k_per_parent`` of the
    parent's samples are promoted from forecast to GPU evaluation, with
    the rest deferred. The fixed-budget invariant is one of v3's load-
    bearing properties (spec § invariants).
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    total_budget_steps: int
    batch_size: int
    samples_per_parent: int
    k_per_parent: int
    archive_capacity: int = 1000
    c_puct: float = 1.0
    ranking_rule: RankingRule
