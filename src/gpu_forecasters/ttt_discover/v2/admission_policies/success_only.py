"""Admit only candidates whose evaluation succeeded.

This mirrors v1's archive policy: failed children (compile / runtime /
incorrect / parse / infra) do not enter the live search tree, so they
cannot be selected as parents in later rollouts. They are still written
to ``rollouts.jsonl`` by the sink — only their presence in the search
state is elided.
"""

from __future__ import annotations

from arid_badger.trimul.core import SuccessFeedback
from arid_badger.ttt_discover.v2.domain.admission_decision import (
    AdmissionDecision,
    AdmitChild,
    CreditOnly,
)
from arid_badger.ttt_discover.v2.domain.candidate import Candidate
from arid_badger.ttt_discover.v2.interfaces.admission_policy import AdmissionPolicy
from arid_badger.typing_utils import implements


class SuccessOnlyAdmissionPolicy:
    def decide(
        self, candidate: Candidate, parent: Candidate | None
    ) -> AdmissionDecision:
        del parent  # unused: decision depends only on the candidate's outcome
        if isinstance(candidate.outcome, SuccessFeedback):
            return AdmitChild()
        return CreditOnly()


_ = implements(AdmissionPolicy)(SuccessOnlyAdmissionPolicy)
