"""File-backed PUCT candidate archive.

Each call to ``snapshot(step)`` writes ``puct_archive_step_{step:06d}.json``
alongside ``self._dir``. The schema is intentionally independent of
v1's sampler file format — this archive stores ``Candidate`` objects
directly (code + outcome + reward) rather than v1's ``State`` /
``construction`` pair, so the two schemas cannot be reused.

``sample(n)`` delegates to ``puct_math.compute_scores`` and returns the
top-``n`` candidates by selection score, suppressing any candidate whose
full lineage (ancestors *and* descendants) overlaps an already-picked
candidate's lineage — identical to v1's "blocked_ids" behaviour so two
sampled rollouts cannot both descend from the same branch in a batch.
"""

from __future__ import annotations

import json
import threading
import uuid
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from arid_badger.trimul.core import (
    CompileFailedFeedback,
    IncorrectFeedback,
    InfrastructureFailureFeedback,
    RuntimeErrorFeedback,
    SuccessFeedback,
)
from arid_badger.ttt_discover.v2.archive.puct_math import (
    compute_prior,
    compute_scale,
    compute_scores,
)
from arid_badger.ttt_discover.v2.domain.candidate import Candidate, CandidateId
from arid_badger.ttt_discover.v2.domain.outcome import (
    ParseFailureFeedback,
    TriMulRLOutcome,
)
from arid_badger.ttt_discover.v2.interfaces.archive import CandidateArchive
from arid_badger.typing_utils import implements


_OUTCOME_MODELS: dict[str, type[BaseModel]] = {
    "compile_failed": CompileFailedFeedback,
    "runtime_error": RuntimeErrorFeedback,
    "incorrect": IncorrectFeedback,
    "success": SuccessFeedback,
    "infrastructure_failure": InfrastructureFailureFeedback,
    "parse_failure": ParseFailureFeedback,
}


def _outcome_from_dict(d: dict[str, Any] | None) -> TriMulRLOutcome | None:
    if d is None:
        return None
    cls = _OUTCOME_MODELS[d["kind"]]
    return cls.model_validate(d)  # pyright: ignore[reportReturnType]


def _outcome_to_dict(outcome: TriMulRLOutcome | None) -> dict[str, Any] | None:
    if outcome is None:
        return None
    return outcome.model_dump()


class _ArchiveSnapshot(BaseModel):
    """On-disk schema for ``puct_archive_step_NNNNNN.json``."""

    model_config = ConfigDict(frozen=True)

    step: int
    candidates: list[dict[str, Any]]
    visits: dict[str, int]
    best_value: dict[str, float]
    total_visits: int


def _new_candidate_id() -> CandidateId:
    return CandidateId(str(uuid.uuid4()))


class PUCTCandidateArchive:
    """In-memory PUCT archive with an on-disk snapshot per training step."""

    _dir: Path
    _puct_c: float
    _candidates: list[Candidate]
    _by_id: dict[CandidateId, Candidate]
    _children_by_parent: dict[CandidateId, set[CandidateId]]
    _visits: dict[CandidateId, int]
    _best_value: dict[CandidateId, float]
    _total_visits: int
    _lock: threading.Lock

    def __deepcopy__(self, memo: dict[int, object]) -> "PUCTCandidateArchive":
        # The archive is shared mutable state — hparam logging deepcopies
        # the Config for serialisation, and the underlying ``threading.Lock``
        # is not pickleable. Returning self preserves identity and avoids
        # the copy path entirely.
        return self

    def __init__(self, directory: Path, *, puct_c: float = 1.0) -> None:
        self._dir = directory
        self._puct_c = puct_c
        self._candidates = []
        self._by_id = {}
        self._children_by_parent = {}
        self._visits = {}
        self._best_value = {}
        self._total_visits = 0
        self._lock = threading.Lock()

        root = Candidate(
            id=_new_candidate_id(),
            code="",
            timestep=-1,
            parent_id=None,
            outcome=None,
            reward=0.0,
        )
        self._register(root)

    def _register(self, candidate: Candidate) -> None:
        self._candidates.append(candidate)
        self._by_id[candidate.id] = candidate
        if candidate.parent_id is not None:
            self._children_by_parent.setdefault(candidate.parent_id, set()).add(
                candidate.id
            )

    def _lineage(self, cid: CandidateId) -> set[CandidateId]:
        """All ancestors + all descendants of ``cid`` (inclusive)."""
        lineage: set[CandidateId] = {cid}
        # Ancestors
        cursor: CandidateId | None = cid
        while cursor is not None:
            cand = self._by_id.get(cursor)
            if cand is None or cand.parent_id is None:
                break
            lineage.add(cand.parent_id)
            cursor = cand.parent_id
        # Descendants (BFS)
        frontier: list[CandidateId] = [cid]
        while frontier:
            current = frontier.pop(0)
            for child_id in self._children_by_parent.get(current, set()):
                if child_id not in lineage:
                    lineage.add(child_id)
                    frontier.append(child_id)
        return lineage

    def sample(self, n: int) -> list[Candidate]:
        if n <= 0:
            return []
        with self._lock:
            if not self._candidates:
                return []
            rewards = np.array([c.reward for c in self._candidates], dtype=np.float64)
            priors = compute_prior(rewards)
            scale = compute_scale(rewards)
            visits_arr = np.array(
                [self._visits.get(c.id, 0) for c in self._candidates],
                dtype=np.float64,
            )
            best_arr = np.array(
                [self._best_value.get(c.id, c.reward) for c in self._candidates],
                dtype=np.float64,
            )
            scores = compute_scores(
                rewards=rewards,
                priors=priors,
                n=visits_arr,
                m=best_arr,
                total_visits=self._total_visits,
                scale=scale,
                puct_c=self._puct_c,
            )
            order = np.argsort(-scores, kind="stable")
            picked: list[Candidate] = []
            blocked: set[CandidateId] = set()
            for idx in order:
                cand = self._candidates[int(idx)]
                if cand.id in blocked:
                    continue
                picked.append(cand)
                blocked.update(self._lineage(cand.id))
                if len(picked) >= n:
                    break
            return picked

    def credit_rollout(
        self, *, parent: Candidate | None, child: Candidate | None
    ) -> None:
        """Account for one completed rollout.

        If ``child`` is provided, it is registered as a new node in the
        tree and the parent's ``best_value`` is updated. In either case
        (child or no child), the parent's subtree visit counts and the
        global expansion counter advance by exactly one. When
        ``parent is None`` (cold-start rollout), there is no subtree to
        credit and the visit-count walk is skipped; the child, if any,
        is still registered.
        """
        with self._lock:
            if child is not None and child.id not in self._by_id:
                self._register(child)
            if parent is None:
                return
            if child is not None:
                self._best_value[parent.id] = max(
                    self._best_value.get(parent.id, parent.reward),
                    child.reward,
                )
            cursor: CandidateId | None = parent.id
            while cursor is not None:
                self._visits[cursor] = self._visits.get(cursor, 0) + 1
                cand = self._by_id.get(cursor)
                cursor = cand.parent_id if cand is not None else None
            self._total_visits += 1

    def snapshot(self, step: int) -> None:
        with self._lock:
            candidates_payload: list[dict[str, Any]] = [
                {
                    "id": c.id,
                    "code": c.code,
                    "timestep": c.timestep,
                    "parent_id": c.parent_id,
                    "outcome": _outcome_to_dict(c.outcome),
                    "reward": c.reward,
                }
                for c in self._candidates
            ]
            snapshot = _ArchiveSnapshot(
                step=step,
                candidates=candidates_payload,
                visits={str(k): int(v) for k, v in self._visits.items()},
                best_value={str(k): float(v) for k, v in self._best_value.items()},
                total_visits=self._total_visits,
            )
            self._dir.mkdir(parents=True, exist_ok=True)
            path = self._dir / f"puct_archive_step_{step:06d}.json"
            tmp = path.with_suffix(".json.tmp")
            _ = tmp.write_text(snapshot.model_dump_json(indent=2))
            _ = tmp.replace(path)

    @classmethod
    def from_snapshot(
        cls, directory: Path, snapshot_path: Path, *, puct_c: float = 1.0
    ) -> "PUCTCandidateArchive":
        raw = json.loads(snapshot_path.read_text())
        snapshot = _ArchiveSnapshot.model_validate(raw)
        archive = cls.__new__(cls)
        archive._dir = directory
        archive._puct_c = puct_c
        archive._candidates = []
        archive._by_id = {}
        archive._children_by_parent = {}
        archive._visits = {
            CandidateId(k): int(v) for k, v in snapshot.visits.items()
        }
        archive._best_value = {
            CandidateId(k): float(v) for k, v in snapshot.best_value.items()
        }
        archive._total_visits = snapshot.total_visits
        archive._lock = threading.Lock()
        for c in snapshot.candidates:
            cand = Candidate(
                id=CandidateId(c["id"]),
                code=c["code"],
                timestep=int(c["timestep"]),
                parent_id=(
                    CandidateId(c["parent_id"]) if c.get("parent_id") else None
                ),
                outcome=_outcome_from_dict(c.get("outcome")),
                reward=float(c["reward"]),
            )
            archive._register(cand)
        return archive


_ = implements(CandidateArchive)(PUCTCandidateArchive)


def build_candidate(
    *,
    code: str,
    timestep: int,
    parent_id: CandidateId | None,
    outcome: TriMulRLOutcome,
    reward: float,
) -> Candidate:
    """Convenience factory — generates a fresh ``CandidateId``."""
    return Candidate(
        id=_new_candidate_id(),
        code=code,
        timestep=timestep,
        parent_id=parent_id,
        outcome=outcome,
        reward=reward,
    )
