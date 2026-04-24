"""A candidate — one node in the PUCT search tree."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NewType

from arid_badger.ttt_discover.v2.domain.outcome import TriMulRLOutcome

CandidateId = NewType("CandidateId", str)


@dataclass(frozen=True)
class Candidate:
    """One kernel + its evaluation outcome, identified by a ``CandidateId``.

    ``outcome`` is ``None`` only for the cold-start root candidate (no
    code yet, no evaluation). ``reward`` is the pre-computed scalarizer
    output for that outcome — cached on the candidate so the archive and
    PUCT math can work with floats without re-scalarising each time.
    """

    id: CandidateId
    code: str
    timestep: int
    parent_id: CandidateId | None
    outcome: TriMulRLOutcome | None
    reward: float
