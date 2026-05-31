"""Concrete ``RiskFunction`` implementations.

Three risks the user agreed to track:

* ``BinaryMismatchRisk`` — pointwise, mean of ``I(predicted_bin != true_bin)``.
* ``SpeedupDistanceRisk`` — pointwise, mean of
  ``|midpoint(predicted_bin) - true_speedup|``. The bin midpoint
  recipe matches ``experiments/e0122_score_zai9_surrogates_canonical/headline_metrics.py``.
* ``RegretRisk`` — set-level. After the user trusts the model on
  predicted rows (selecting the model's overall top-bin entries among
  them) and real-evaluates the abstained ones, what speedup do they
  fail to recover relative to the pack's true best?
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from typing import Sequence

from gpu_forecasters.abstaining_evaluation.v1.domain import (
    AbstainDecision,
    Predict,
    PredictOrAbstain,
    RiskFunction,
)
from gpu_forecasters.eval_dataset_builder.v1 import KernelRuntimeComparison
from gpu_forecasters.landscape_map.v2 import SpeedupBin
from gpu_forecasters.typing_utils import implements


_LOG2_BIN_CENTERS: dict[int, float] = {
    2: -1.5,
    3: -0.75,
    4: -0.25,
    5: 0.25,
    6: 0.75,
    7: 1.5,
}


def bin_midpoint(
    b: SpeedupBin,
    *,
    bin_1_empirical: float,
    bin_8_empirical: float,
) -> float:
    """Midpoint of a speedup bin, in speedup units.

    Bins 2-7 use the geometric midpoint of the bin's interval. Bins 1
    and 8 are open-ended on one side, so we use the empirical mean of
    true speedups within that bin (computed from the eval-set in
    use). ``FAILURE`` (bin 0) is mapped to 0.0, matching e0122's
    behavior so a FAILURE forecast on a fast kernel pays full L1
    distance.
    """
    bi = int(b)
    if bi == 0:
        return 0.0
    if bi == 1:
        return bin_1_empirical
    if bi == 8:
        return bin_8_empirical
    return 2.0 ** _LOG2_BIN_CENTERS[bi]


def _empirical_open_bin_midpoints(
    comparisons: Sequence[KernelRuntimeComparison],
) -> tuple[float, float]:
    """``(bin_1_mean, bin_8_mean)`` empirical midpoints for the given
    rows, with the same fallbacks e0122 uses when the bin is unrepresented.
    """
    bin_1 = [
        c.aggregated_speedup for c in comparisons if int(c.true_bin) == 1
    ]
    bin_8 = [
        c.aggregated_speedup for c in comparisons if int(c.true_bin) == 8
    ]
    bin_1_mid = statistics.fmean(bin_1) if bin_1 else 0.125
    bin_8_mid = statistics.fmean(bin_8) if bin_8 else 4.0 * (2.0 ** 0.5)
    return bin_1_mid, bin_8_mid


def _predicted_pairs(
    decisions: Sequence[PredictOrAbstain],
    comparisons: Sequence[KernelRuntimeComparison],
) -> list[tuple[Predict, KernelRuntimeComparison]]:
    """Filter to the predicted subset, paired by row."""
    if len(decisions) != len(comparisons):
        raise ValueError(
            f"decisions and comparisons must align "
            f"(got {len(decisions)} vs {len(comparisons)})"
        )
    return [
        (d, c)
        for d, c in zip(decisions, comparisons)
        if isinstance(d, Predict)
    ]


@dataclass(frozen=True)
class BinaryMismatchRisk:
    """Mean of ``I(predicted_bin != true_bin)`` over the predicted subset.

    Returns NaN when the predicted subset is empty (no coverage).
    Callers reading ``risk_coverage_curve`` output may want to drop
    NaN points before integrating.
    """

    name: str = "binary_mismatch"

    def __call__(
        self,
        decisions: Sequence[PredictOrAbstain],
        comparisons: Sequence[KernelRuntimeComparison],
    ) -> float:
        pairs = _predicted_pairs(decisions, comparisons)
        if not pairs:
            return float("nan")
        # ``KernelRuntimeComparison.true_bin`` is the v1 SpeedupBin enum;
        # ``KernelRuntimeEstimate.predicted_bin`` is v2's. They share
        # underlying ints, so compare via int.
        wrong = sum(
            1
            for d, c in pairs
            if int(d.estimate.predicted_bin) != int(c.true_bin)
        )
        return wrong / len(pairs)


@dataclass(frozen=True)
class SpeedupDistanceRisk:
    """Mean of ``|midpoint(predicted_bin) - true_speedup|``.

    Bin midpoints follow ``bin_midpoint`` (geometric for bins 2-7,
    empirical for bins 1 and 8). The empirical midpoints are computed
    once per call from the *full* ``comparisons`` argument so the
    midpoint is stable as coverage changes.
    """

    name: str = "speedup_distance"

    def __call__(
        self,
        decisions: Sequence[PredictOrAbstain],
        comparisons: Sequence[KernelRuntimeComparison],
    ) -> float:
        bin_1_mid, bin_8_mid = _empirical_open_bin_midpoints(comparisons)
        pairs = _predicted_pairs(decisions, comparisons)
        if not pairs:
            return float("nan")
        distances = [
            abs(
                bin_midpoint(
                    d.estimate.predicted_bin,
                    bin_1_empirical=bin_1_mid,
                    bin_8_empirical=bin_8_mid,
                )
                - c.aggregated_speedup
            )
            for d, c in pairs
        ]
        return statistics.fmean(distances)


@dataclass(frozen=True)
class RegretRisk:
    """Set-level regret after trusting the model + real-eval on abstain.

    Operational story: at deployment, the user real-evaluates every
    abstained row (so its true speedup is known) and trusts the model
    on predicted rows by taking the model's overall top predicted bin
    and selecting the best among predicted candidates assigned to it.
    Their realized best speedup is

        max(
            max true speedup among abstained,
            max true speedup among predicted candidates whose predicted
            bin equals max(predicted_bin) over the predicted subset,
        )

    Regret is ``pack_best - realized_best`` in speedup units. Lower is
    better; zero is unrecoverable from any policy that's allowed to
    real-eval the abstained subset. When the predicted subset is
    empty, regret is zero (everything was real-evaluated).
    """

    name: str = "regret"

    def __call__(
        self,
        decisions: Sequence[PredictOrAbstain],
        comparisons: Sequence[KernelRuntimeComparison],
    ) -> float:
        if len(decisions) != len(comparisons):
            raise ValueError(
                f"decisions and comparisons must align "
                f"(got {len(decisions)} vs {len(comparisons)})"
            )
        if not comparisons:
            return float("nan")
        pack_best = max(c.aggregated_speedup for c in comparisons)

        predicted_pairs = [
            (d, c)
            for d, c in zip(decisions, comparisons)
            if isinstance(d, Predict)
        ]
        abstained_speedups = [
            c.aggregated_speedup
            for d, c in zip(decisions, comparisons)
            if isinstance(d, AbstainDecision)
        ]

        if not predicted_pairs:
            # Everything was abstained on; user gets ground truth back
            # for every row.
            best_abstained = (
                max(abstained_speedups) if abstained_speedups else 0.0
            )
            return pack_best - best_abstained

        top_bin = max(
            int(d.estimate.predicted_bin) for d, _ in predicted_pairs
        )
        top_bin_speedups = [
            c.aggregated_speedup
            for d, c in predicted_pairs
            if int(d.estimate.predicted_bin) == top_bin
        ]
        if not top_bin_speedups:
            best_predicted = float("-inf")
        else:
            best_predicted = max(top_bin_speedups)

        best_abstained = (
            max(abstained_speedups) if abstained_speedups else float("-inf")
        )
        realized_best = max(best_predicted, best_abstained)
        return pack_best - realized_best


implements(RiskFunction)(BinaryMismatchRisk)
implements(RiskFunction)(SpeedupDistanceRisk)
implements(RiskFunction)(RegretRisk)
