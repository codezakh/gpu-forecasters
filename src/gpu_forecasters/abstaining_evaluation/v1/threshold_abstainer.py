"""Threshold-based abstain policy parameterized by a confidence score.

The single concrete ``AbstainPolicy`` we ship in v1. Native-abstain
(an LLM that picks the abstain decision itself) is not a policy in
this sense — it produces decisions directly without a row-level score.
"""

from __future__ import annotations

from dataclasses import dataclass

from gpu_forecasters.abstaining_evaluation.v1.domain import (
    AbstainDecision,
    AbstainPolicy,
    ConfidenceScore,
    Predict,
    PredictOrAbstain,
)
from gpu_forecasters.landscape_map.v2 import KernelRuntimeEstimate
from gpu_forecasters.typing_utils import implements


@dataclass(frozen=True)
class ThresholdAbstainPolicy:
    """Abstain when the score falls below a threshold.

    Parse failures (``estimate is None`` — the surrogate did not
    produce a parseable tool call) are forced abstentions regardless
    of threshold; they decrement coverage but contribute their true
    speedup to the abstained subset for the regret risk.
    """

    score: ConfidenceScore
    threshold: float

    def __call__(
        self, estimate: KernelRuntimeEstimate | None
    ) -> PredictOrAbstain:
        if estimate is None:
            return AbstainDecision(reason="parse_failure")
        if self.score(estimate) < self.threshold:
            return AbstainDecision()
        return Predict(estimate=estimate)


implements(AbstainPolicy)(ThresholdAbstainPolicy)
