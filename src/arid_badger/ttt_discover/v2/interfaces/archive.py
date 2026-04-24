from __future__ import annotations

from typing import Protocol

from arid_badger.ttt_discover.v2.domain.candidate import Candidate


class CandidateArchive(Protocol):
    """The live search state shared across rollouts in a run.

    ``sample`` returns ``n`` parent candidates for the next batch;
    ``insert`` adds a new child keyed off ``parent``; ``record_failed_
    attempt`` updates visit counts when a rollout produced nothing usable
    (parse / compile / runtime / infra failures) so the PUCT math can
    still back-propagate. ``snapshot`` persists the current archive
    contents under a step-indexed filename; callers invoke it each
    training step.
    """

    def sample(self, n: int) -> list[Candidate]: ...

    def insert(self, candidate: Candidate, parent: Candidate | None) -> None: ...

    def record_failed_attempt(self, parent: Candidate | None) -> None: ...

    def snapshot(self, step: int) -> None: ...
