"""Policy deciding whether a candidate joins the live search tree.

Pure function of the candidate and its parent: ``decide`` inspects the
rollout outcome on the candidate and returns a typed
``AdmissionDecision``. The env applies that decision to the archive via
``credit_rollout``.

Concretes live in ``arid_badger.ttt_discover.v2.admission_policies``.
"""

from __future__ import annotations

from typing import Protocol

from arid_badger.ttt_discover.v2.domain.admission_decision import AdmissionDecision
from arid_badger.ttt_discover.v2.domain.candidate import Candidate


class AdmissionPolicy(Protocol):
    def decide(
        self, candidate: Candidate, parent: Candidate | None
    ) -> AdmissionDecision: ...
