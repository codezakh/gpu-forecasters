"""Compare a candidate TriMul kernel against cached top-leaderboard kernels.

Public entrypoint is :func:`compare_to_leaderboard`. The 5 leaderboard
fixtures in ``tests/trimul/fixtures/leaderboard/`` are pre-scored once per
GPU via :func:`bootstrap_leaderboard_baselines` and stored as JSON next to
this module under ``_leaderboard_baselines/{gpu}/{kernel_id}.json``.

Container-to-container timing variance affects cached and freshly-measured
runs symmetrically (each Modal scoring call gets its own container regardless
of whether kernels are dispatched in the same session), so caching adds no
asymmetric error. The longer time axis for hardware/driver drift is
mitigated by the ``measured_at`` timestamp on every scorecard.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import math
import warnings
from enum import StrEnum
from pathlib import Path
from typing import Iterable

from pydantic import BaseModel, ConfigDict

from arid_badger.trimul.cases import BENCHMARK_CASES, TriMulTestArgs
from arid_badger.trimul.modal_scoring import modal_trimul_scoring_session
from arid_badger.typing_utils import is_ok


# ---------------------------------------------------------------------------
# Enums and registry
# ---------------------------------------------------------------------------


class GpuKind(StrEnum):
    """Modal GPU identifiers. String values match Modal's ``gpu=`` argument
    exactly so they pass through ``.with_options(gpu=...)`` unchanged."""

    A100_80GB = "A100-80GB"


class LeaderboardKernelId(StrEnum):
    """Stable short identifiers for the cached leaderboard kernels.
    These are the cache keys: baseline JSON filenames are ``{id.value}.json``."""

    TTT = "ttt"
    SHIYEGAO = "shiyegao"
    EMMETT_BICKER = "emmett_bicker"
    ARSENI_IVANOV = "arseni_ivanov"
    WAQAR = "waqar"


class LeaderboardKernelEntry(BaseModel):
    """One row of the leaderboard registry."""

    model_config = ConfigDict(frozen=True)

    fixture_filename: str
    display_name: str
    published_geomean_us: float


# Canonical registry. In published-rank order (TTT #1 → Waqar #5 on A100).
# Replaces the duplicate _LEADERBOARD_KERNELS list previously living in
# tests/trimul/test_modal_scoring_integration.py.
LEADERBOARD_REGISTRY: dict[LeaderboardKernelId, LeaderboardKernelEntry] = {
    LeaderboardKernelId.TTT: LeaderboardKernelEntry(
        fixture_filename="ttt-discover.py",
        display_name="TTT",
        published_geomean_us=2198.190,
    ),
    LeaderboardKernelId.SHIYEGAO: LeaderboardKernelEntry(
        fixture_filename="shiyegao.py",
        display_name="shiyegao",
        published_geomean_us=2273.0,
    ),
    LeaderboardKernelId.EMMETT_BICKER: LeaderboardKernelEntry(
        fixture_filename="emmet-bicker.py",
        display_name="Emmett Bicker",
        published_geomean_us=2370.0,
    ),
    LeaderboardKernelId.ARSENI_IVANOV: LeaderboardKernelEntry(
        fixture_filename="areseni_ivanov.py",
        display_name="Arseni Ivanov",
        published_geomean_us=4532.0,
    ),
    LeaderboardKernelId.WAQAR: LeaderboardKernelEntry(
        fixture_filename="waqar.py",
        display_name="Waqar",
        published_geomean_us=4919.0,
    ),
}


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


class CaseResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_index: int
    nomask: bool
    correct: bool
    runtime_ns: float | None
    failure_kind: str | None  # None when correct


class KernelScorecard(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    leaderboard_id: LeaderboardKernelId | None  # None for the candidate
    source_sha256: str  # drift detector; not a key
    measured_at: str  # ISO-8601 UTC
    published_geomean_us: float | None  # None for candidate
    case_results: list[CaseResult]


class LeaderboardComparison(BaseModel):
    model_config = ConfigDict(frozen=True)

    gpu: GpuKind
    candidate_scorecard: KernelScorecard
    leaderboard_scorecards: list[KernelScorecard]  # in published-rank order

    def ranking_all_cases(self) -> list[tuple[str, float]]:
        """Geomean over all BENCHMARK_CASES. Excludes any kernel incorrect on
        any case (see _LEADERBOARD_KNOWN_BAD note in test_modal_scoring_integration.py
        for why TTT drops out of this view by design)."""
        return self._ranking(case_indices=range(len(BENCHMARK_CASES)))

    def ranking_nomask_true(self) -> list[tuple[str, float]]:
        """Geomean over the nomask=True subset of BENCHMARK_CASES. Same
        exclusion rule restricted to this subset — keeps TTT in the comparison."""
        nomask_indices = [
            i for i, case in enumerate(BENCHMARK_CASES) if case["nomask"]
        ]
        return self._ranking(case_indices=nomask_indices)

    def render_tables(self) -> str:
        return _render_tables(self)

    def _all_scorecards(self) -> list[KernelScorecard]:
        return [self.candidate_scorecard, *self.leaderboard_scorecards]

    def _ranking(self, *, case_indices: Iterable[int]) -> list[tuple[str, float]]:
        idxs = list(case_indices)
        ranked: list[tuple[str, float]] = []
        for sc in self._all_scorecards():
            relevant = [r for r in sc.case_results if r.case_index in idxs]
            if not all(r.correct and r.runtime_ns is not None for r in relevant):
                continue
            runtimes = [r.runtime_ns for r in relevant]
            assert all(rt is not None for rt in runtimes)
            geo = math.exp(
                sum(math.log(rt) for rt in runtimes if rt is not None) / len(runtimes)
            )
            ranked.append((sc.name, geo))
        ranked.sort(key=lambda x: x[1])
        return ranked


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------


_BASELINES_ROOT = Path(__file__).parent / "_leaderboard_baselines"
_FIXTURES_ROOT = (
    Path(__file__).resolve().parents[3]  # src/arid_badger/trimul → repo root of 15-arid-badger
    / "tests"
    / "trimul"
    / "fixtures"
    / "leaderboard"
)


def _baseline_path(gpu: GpuKind, kernel_id: LeaderboardKernelId) -> Path:
    return _BASELINES_ROOT / gpu.value / f"{kernel_id.value}.json"


def _fixture_source(kernel_id: LeaderboardKernelId) -> str:
    entry = LEADERBOARD_REGISTRY[kernel_id]
    return (_FIXTURES_ROOT / entry.fixture_filename).read_text()


def _sha256(source: str) -> str:
    return hashlib.sha256(source.encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Scoring → scorecard
# ---------------------------------------------------------------------------


def _score_to_scorecard(
    *,
    name: str,
    leaderboard_id: LeaderboardKernelId | None,
    source: str,
    score_fn,
    published_geomean_us: float | None,
) -> KernelScorecard:
    """Run ``source`` through ``score_fn`` over BENCHMARK_CASES, build a scorecard."""
    cases: list[TriMulTestArgs] = list(BENCHMARK_CASES)
    results = score_fn(source, cases)
    assert len(results) == len(cases), (
        f"scoring returned {len(results)} results for {len(cases)} cases"
    )
    case_results: list[CaseResult] = []
    for idx, (case, result) in enumerate(zip(cases, results)):
        if not is_ok(result):
            case_results.append(
                CaseResult(
                    case_index=idx,
                    nomask=case["nomask"],
                    correct=False,
                    runtime_ns=None,
                    failure_kind="infrastructure_failure",
                )
            )
            continue
        exec_result = result.unwrap()
        if exec_result.correct:
            case_results.append(
                CaseResult(
                    case_index=idx,
                    nomask=case["nomask"],
                    correct=True,
                    runtime_ns=exec_result.runtime_ns,
                    failure_kind=None,
                )
            )
        else:
            case_results.append(
                CaseResult(
                    case_index=idx,
                    nomask=case["nomask"],
                    correct=False,
                    runtime_ns=None,
                    failure_kind=exec_result.failure_kind,
                )
            )
    return KernelScorecard(
        name=name,
        leaderboard_id=leaderboard_id,
        source_sha256=_sha256(source),
        measured_at=_now_iso(),
        published_geomean_us=published_geomean_us,
        case_results=case_results,
    )


# ---------------------------------------------------------------------------
# Cache I/O
# ---------------------------------------------------------------------------


def load_leaderboard_baseline(
    gpu: GpuKind,
) -> dict[LeaderboardKernelId, KernelScorecard]:
    """Load every cached leaderboard scorecard for ``gpu``.

    Raises FileNotFoundError if any registry kernel is missing a baseline.
    Emits a warning per scorecard whose recorded ``source_sha256`` no longer
    matches the on-disk fixture (i.e. someone edited a fixture and the
    baseline is now stale)."""
    out: dict[LeaderboardKernelId, KernelScorecard] = {}
    missing: list[str] = []
    for kid in LEADERBOARD_REGISTRY:
        path = _baseline_path(gpu, kid)
        if not path.exists():
            missing.append(str(path))
            continue
        scorecard = KernelScorecard.model_validate_json(path.read_text())
        current_sha = _sha256(_fixture_source(kid))
        if scorecard.source_sha256 != current_sha:
            warnings.warn(
                f"Baseline for {kid.value} on {gpu.value} is stale: "
                f"recorded sha {scorecard.source_sha256[:12]} != current "
                f"fixture sha {current_sha[:12]}. Re-run "
                f"`bootstrap_leaderboard_baselines([GpuKind.{gpu.name}])`.",
                stacklevel=2,
            )
        out[kid] = scorecard
    if missing:
        raise FileNotFoundError(
            f"No cached baselines for GPU {gpu.value}: missing files {missing}. "
            f"Run `bootstrap-trimul-leaderboard --gpu {gpu.value}` first."
        )
    return out


def _write_baseline(scorecard: KernelScorecard, gpu: GpuKind) -> None:
    assert scorecard.leaderboard_id is not None, (
        "only leaderboard scorecards are cached"
    )
    path = _baseline_path(gpu, scorecard.leaderboard_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(scorecard.model_dump_json(indent=2))
    tmp.replace(path)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def compare_to_leaderboard(
    candidate_source: str,
    gpu: GpuKind,
    *,
    candidate_name: str = "candidate",
) -> LeaderboardComparison:
    """Score ``candidate_source`` on ``gpu`` over BENCHMARK_CASES, combine
    with cached leaderboard baselines for ``gpu``, return the comparison."""
    baselines = load_leaderboard_baseline(gpu)
    with modal_trimul_scoring_session(gpu=gpu.value) as score:
        candidate_scorecard = _score_to_scorecard(
            name=candidate_name,
            leaderboard_id=None,
            source=candidate_source,
            score_fn=score,
            published_geomean_us=None,
        )
    leaderboard_scorecards = [baselines[kid] for kid in LEADERBOARD_REGISTRY]
    return LeaderboardComparison(
        gpu=gpu,
        candidate_scorecard=candidate_scorecard,
        leaderboard_scorecards=leaderboard_scorecards,
    )


def bootstrap_leaderboard_baselines(gpus: list[GpuKind]) -> None:
    """Score every leaderboard fixture on every GPU and write baseline JSONs.
    Idempotent: existing baselines are overwritten."""
    for gpu in gpus:
        with modal_trimul_scoring_session(gpu=gpu.value) as score:
            for kid, entry in LEADERBOARD_REGISTRY.items():
                source = _fixture_source(kid)
                scorecard = _score_to_scorecard(
                    name=entry.display_name,
                    leaderboard_id=kid,
                    source=source,
                    score_fn=score,
                    published_geomean_us=entry.published_geomean_us,
                )
                _write_baseline(scorecard, gpu)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


def _render_one_table(
    title: str,
    ranking: list[tuple[str, float]],
    all_names: list[str],
) -> str:
    by_name = dict(ranking)
    lines = [title, "-" * len(title)]
    name_w = max(len(n) for n in all_names)
    for rank, (name, geo_ns) in enumerate(ranking, start=1):
        lines.append(f"  {rank}. {name:<{name_w}}  {geo_ns / 1000:>10.1f} μs")
    excluded = [n for n in all_names if n not in by_name]
    for name in excluded:
        lines.append(f"  --  {name:<{name_w}}  {'INCORRECT':>10}")
    return "\n".join(lines)


def _render_tables(cmp: LeaderboardComparison) -> str:
    all_names = [sc.name for sc in cmp._all_scorecards()]
    parts = [
        f"Leaderboard comparison on {cmp.gpu.value}",
        f"Candidate: {cmp.candidate_scorecard.name}",
        "",
        _render_one_table(
            "All 7 BENCHMARK_CASES (geomean)",
            cmp.ranking_all_cases(),
            all_names,
        ),
        "",
        _render_one_table(
            "5 nomask=True BENCHMARK_CASES (geomean)",
            cmp.ranking_nomask_true(),
            all_names,
        ),
    ]
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli() -> None:
    parser = argparse.ArgumentParser(
        prog="bootstrap-trimul-leaderboard",
        description=(
            "Score every leaderboard fixture on each --gpu via Modal and write "
            "baseline JSONs into _leaderboard_baselines/{gpu}/. Idempotent: "
            "existing baselines are overwritten. Required before "
            "compare_to_leaderboard() can be used on a new GPU."
        ),
    )
    parser.add_argument(
        "--gpu",
        action="append",
        required=True,
        choices=[g.value for g in GpuKind],
        help="GPU to bootstrap (repeatable for multiple GPUs in one invocation).",
    )
    args = parser.parse_args()
    gpus = [GpuKind(v) for v in args.gpu]
    print(f"Bootstrapping baselines for: {[g.value for g in gpus]}")
    bootstrap_leaderboard_baselines(gpus)
    print("Done.")


if __name__ == "__main__":
    _cli()
