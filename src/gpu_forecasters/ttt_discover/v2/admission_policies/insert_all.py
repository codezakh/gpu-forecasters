"""Admit every candidate with a completed outcome.

Failed children (compile / runtime / incorrect / parse) enter the tree
as reward-0 nodes. PUCT's rank-based prior places them below successes;
the exploration bonus still gives them occasional selection, which is
how a failure gets a chance to surface as a parent in a subsequent
rollout — closing the loop the feedback renderer's failure-variant
branches were written for.

Infrastructure failures are still elided because they are not the
model's fault and carry no actionable signal for a future rollout.
Cold-start candidates with ``outcome is None`` (i.e. the root) are
never produced by the env, so the policy does not need to consider that
case.
"""

from __future__ import annotations

from gpu_forecasters.trimul.core import InfrastructureFailureFeedback
from gpu_forecasters.ttt_discover.v2.domain.admission_decision import (
    AdmissionDecision,
    AdmitChild,
    CreditOnly,
)
from gpu_forecasters.ttt_discover.v2.domain.candidate import Candidate
from gpu_forecasters.ttt_discover.v2.interfaces.admission_policy import AdmissionPolicy
from gpu_forecasters.typing_utils import implements


class InsertAllAdmissionPolicy:
    def decide(
        self, candidate: Candidate, parent: Candidate | None
    ) -> AdmissionDecision:
        del parent  # unused: decision depends only on the candidate's outcome
        if isinstance(candidate.outcome, InfrastructureFailureFeedback):
            return CreditOnly()
        return AdmitChild()


_ = implements(AdmissionPolicy)(InsertAllAdmissionPolicy)
