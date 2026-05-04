"""Domain types for balanced-pair sampling.

A *candidate* is one correct kernel for a problem with a measured
runtime. A *labeled pair* is an ordered pair `(anchor, candidate)` of
candidates from the same problem, with the candidate's speedup
measured against the anchor: ``S(candidate|anchor) = runtime_anchor /
runtime_candidate``. Labels are continuous; the discrete bin is
derived on demand from :class:`SpeedupBin.from_speedup`.

The pairwise relabeling is the augmentation: a single candidate
contributes to many rows under different anchors, so a model cannot
memorize "code -> bin" — it has to compute the relationship.
"""

from __future__ import annotations

from typing import NewType

from pydantic import BaseModel


# A stable identifier for the problem / task / pack the candidate belongs
# to. The sampler keys all per-problem accounting on this string; callers
# pick the convention (e.g. pack name for six-pack, "L<level>/T<task>/<op>"
# for Sakana).
ProblemId = NewType("ProblemId", str)


class CandidateKernel(BaseModel, frozen=True):
    """One correct kernel implementation for a problem, with runtime.

    ``runtime`` is in arbitrary units — only ratios within a problem
    are used. Different problems may have wildly different scales.
    """

    problem_id: ProblemId
    code_hash: str
    code: str
    runtime: float


class LabeledPair(BaseModel, frozen=True):
    """An ordered (anchor, candidate) pair with a continuous speedup label.

    ``log2_speedup`` = log2(runtime_anchor / runtime_candidate). Bin is
    not stored; it's derived via :class:`SpeedupBin.from_speedup` so a
    future re-binning doesn't invalidate persisted rows.
    """

    problem_id: ProblemId
    anchor_code_hash: str
    anchor_code: str
    candidate_code_hash: str
    candidate_code: str
    log2_speedup: float
