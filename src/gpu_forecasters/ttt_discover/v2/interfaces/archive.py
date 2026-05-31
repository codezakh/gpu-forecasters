from __future__ import annotations

from typing import Protocol

from gpu_forecasters.ttt_discover.v2.domain.candidate import Candidate


class CandidateArchive(Protocol):
    """The live search state shared across rollouts in a run.

    The archive exposes two operations:

    - ``sample`` returns ``n`` parent candidates for the next batch.
    - ``credit_rollout`` is the single mutation operation. Every
      completed rollout calls it exactly once, with ``child=candidate``
      to also register the child as a node in the tree, or ``child=None``
      to only account for the rollout in visit-count bookkeeping. In
      either case the parent's subtree visit counts (and the global
      expansion counter) advance by exactly one. This collapses what
      used to be a two-method interface (``insert`` +
      ``record_failed_attempt``) into one operation whose shape makes
      the mutual exclusion invariant — one call per rollout — impossible
      to violate.
    - ``snapshot`` persists the current archive contents under a
      step-indexed filename; callers invoke it each training step.
    """

    def sample(self, n: int) -> list[Candidate]: ...

    def credit_rollout(
        self, *, parent: Candidate | None, child: Candidate | None
    ) -> None: ...

    def snapshot(self, step: int) -> None: ...
