"""Candidate sources — adapters that produce ``CandidateKernel`` records
from a given upstream dataset.

A :class:`CandidateSource` is a pure function that, when called,
returns the full set of correct candidates for some scope (one Sakana
level, one e0121 dataset, etc.). The sampler does not care where
candidates come from; it only cares that they have problem ids and
runtimes.

Sources should not perform sampling themselves — sampling and
balancing is the sampler's job. A source's output is the raw pool.
"""

from __future__ import annotations

import hashlib
from typing import Protocol, Sequence

from arid_badger.datasets.sakana_archive import load_archive_rows

from .domain import CandidateKernel, ProblemId


def _sha16(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


class CandidateSource(Protocol):
    """A producer of correct-kernel candidates for some scope."""

    def candidates(self) -> list[CandidateKernel]: ...


class SakanaCandidateSource:
    """All correct candidates from one or more Sakana archive levels.

    Each candidate's ``runtime`` is its measured ``CUDA_Runtime`` from
    the archive (ms). ``problem_id`` is ``L<level>/T<task>/<op>``.
    """

    _levels: list[int]

    def __init__(self, levels: list[int]) -> None:
        if not levels:
            raise ValueError("levels must be non-empty")
        self._levels = list(levels)

    def candidates(self) -> list[CandidateKernel]:
        rows_by_task = load_archive_rows(self._levels)
        out: list[CandidateKernel] = []
        for (level, task_id, op), rows in rows_by_task.items():
            problem_id = ProblemId(f"L{level}/T{task_id}/{op}")
            for row in rows:
                if not row.get("Correct"):
                    continue
                cuda_runtime = row.get("CUDA_Runtime")
                if cuda_runtime is None:
                    continue
                runtime = float(cuda_runtime)
                if runtime <= 0:
                    continue
                code = str(row.get("CUDA_Code", ""))
                if not code:
                    continue
                out.append(
                    CandidateKernel(
                        problem_id=problem_id,
                        code_hash=_sha16(code),
                        code=code,
                        runtime=runtime,
                    )
                )
        return out


class SeedRelativeRow(Protocol):
    """The shape this source needs from each row: a candidate measured
    against the pack seed, with ``speedup_geomean`` = S(candidate|seed)
    and ``anchor_source`` reporting whether it's a seed-anchored row.

    Fields are read-only via ``@property`` so frozen Pydantic models
    (e.g. e0121's ``LabeledKernel``) satisfy this protocol structurally.
    """

    @property
    def pack_name(self) -> str: ...

    @property
    def anchor_source(self) -> str: ...

    @property
    def candidate_code(self) -> str: ...

    @property
    def speedup_geomean(self) -> float: ...


class GpuModeSeedAnchoredCandidateSource:
    """Distinct successful candidates from a GPU-mode dataset such as
    e0121's ``training_dataset.jsonl``.

    Each distinct successful candidate appears once as an
    ``anchor_source="seed"`` row in the source dataset, recording its
    seed-relative speedup. We invert that to a synthetic runtime
    (``1 / speedup``) so pairwise ratios reproduce the correct
    candidate-vs-candidate speedup. Only ratios are used downstream;
    the absolute scale is unitless.

    Pass any list of rows that has the seed-anchored shape — e.g.
    ``E0121Results.load().rows()`` or any later experiment that
    persists ``LabeledKernel`` rows.
    """

    _rows: Sequence[SeedRelativeRow]

    def __init__(self, rows: Sequence[SeedRelativeRow]) -> None:
        self._rows = rows

    def candidates(self) -> list[CandidateKernel]:
        seen: set[tuple[str, str]] = set()
        out: list[CandidateKernel] = []
        for row in self._rows:
            if row.anchor_source != "seed":
                continue
            ch = _sha16(row.candidate_code)
            key = (row.pack_name, ch)
            if key in seen:
                continue
            seen.add(key)
            # Invert seed-relative speedup to a synthetic runtime; only
            # within-problem ratios matter.
            synth_runtime = 1.0 / row.speedup_geomean
            out.append(
                CandidateKernel(
                    problem_id=ProblemId(row.pack_name),
                    code_hash=ch,
                    code=row.candidate_code,
                    runtime=synth_runtime,
                )
            )
        return out
