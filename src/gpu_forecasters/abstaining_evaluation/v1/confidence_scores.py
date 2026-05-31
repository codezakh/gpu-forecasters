"""Concrete ``ConfidenceScore`` implementations.

Each is a tiny callable rather than a free function so they can carry
a stable ``name`` for table rendering and so a future score that needs
state (calibration table, learned head, ...) plugs into the same seam.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from arid_badger.abstaining_evaluation.v1.domain import ConfidenceScore
from arid_badger.landscape_map.v2 import KernelRuntimeEstimate
from arid_badger.typing_utils import implements


@dataclass(frozen=True)
class MaxProbScore:
    """Confidence = probability mass on the predicted (argmax) bin.

    The textbook ``max_p`` selective-classification baseline. Higher =
    more confident. Range is ``[1/8, 1]`` since the v2 distribution is
    a true simplex over eight bins.
    """

    name: str = "max_prob"

    def __call__(self, estimate: KernelRuntimeEstimate) -> float:
        return estimate.bin_probabilities[estimate.predicted_bin]


_MAX_ENTROPY = math.log(8.0)


@dataclass(frozen=True)
class NegEntropyScore:
    """Confidence = ``1 - H(p)/log(8)``.

    Negative-entropy normalized to ``[0, 1]``: a delta on one bin gives
    1, the uniform distribution gives 0. Sensitive to the *whole*
    distribution rather than only the argmax mass.
    """

    name: str = "neg_entropy"

    def __call__(self, estimate: KernelRuntimeEstimate) -> float:
        h = 0.0
        for p in estimate.bin_probabilities.values():
            if p > 0.0:
                h -= p * math.log(p)
        return 1.0 - h / _MAX_ENTROPY


@dataclass(frozen=True)
class Top2MarginScore:
    """Confidence = ``p(top1) - p(top2)``.

    The two-class margin a calibrated abstainer would use to gate
    decisions when the top two bins are competing. Range is ``[0, 1]``.
    """

    name: str = "top2_margin"

    def __call__(self, estimate: KernelRuntimeEstimate) -> float:
        sorted_p = sorted(estimate.bin_probabilities.values(), reverse=True)
        return sorted_p[0] - sorted_p[1]


implements(ConfidenceScore)(MaxProbScore)
implements(ConfidenceScore)(NegEntropyScore)
implements(ConfidenceScore)(Top2MarginScore)
