"""Tests for leaderboard_comparison.

Pure-logic tests run by default. The Modal-gated integration test at the
bottom of this file is gated by `modal` + `integration` markers and requires
A100-80GB baselines to exist on disk."""

from __future__ import annotations

import math

import pytest

from gpu_forecasters.trimul.cases import BENCHMARK_CASES
from gpu_forecasters.trimul.leaderboard_comparison import (
    CaseResult,
    GpuKind,
    KernelScorecard,
    LeaderboardComparison,
    LeaderboardKernelId,
    _baseline_path,
    compare_to_leaderboard,
    load_leaderboard_baseline,
)


def _make_scorecard(
    *,
    name: str,
    runtime_ns_per_case: list[float | None],
    leaderboard_id: LeaderboardKernelId | None = None,
    published_geomean_us: float | None = None,
) -> KernelScorecard:
    assert len(runtime_ns_per_case) == len(BENCHMARK_CASES)
    case_results: list[CaseResult] = []
    for i, (case, rt) in enumerate(zip(BENCHMARK_CASES, runtime_ns_per_case)):
        if rt is None:
            case_results.append(
                CaseResult(
                    case_index=i,
                    nomask=case["nomask"],
                    correct=False,
                    runtime_ns=None,
                    failure_kind="incorrect",
                )
            )
        else:
            case_results.append(
                CaseResult(
                    case_index=i,
                    nomask=case["nomask"],
                    correct=True,
                    runtime_ns=rt,
                    failure_kind=None,
                )
            )
    return KernelScorecard(
        name=name,
        leaderboard_id=leaderboard_id,
        source_sha256="0" * 64,
        measured_at="2026-04-16T00:00:00+00:00",
        published_geomean_us=published_geomean_us,
        case_results=case_results,
    )


def _make_comparison(
    *,
    candidate: KernelScorecard,
    leaderboard: list[KernelScorecard],
) -> LeaderboardComparison:
    return LeaderboardComparison(
        gpu=GpuKind.A100_80GB,
        candidate_scorecard=candidate,
        leaderboard_scorecards=leaderboard,
    )


def test_geomean_all_cases_uniform_runtime() -> None:
    """A scorecard with the same runtime on every case yields that runtime as geomean."""
    candidate = _make_scorecard(
        name="candidate",
        runtime_ns_per_case=[2_000_000.0] * len(BENCHMARK_CASES),
    )
    cmp = _make_comparison(candidate=candidate, leaderboard=[])
    ranking = cmp.ranking_all_cases()
    assert len(ranking) == 1
    assert ranking[0][0] == "candidate"
    assert math.isclose(ranking[0][1], 2_000_000.0, rel_tol=1e-9)


def test_ranking_all_cases_excludes_kernel_with_incorrect_case() -> None:
    """A kernel incorrect on any case must drop out of the all-cases ranking."""
    candidate = _make_scorecard(
        name="candidate",
        runtime_ns_per_case=[1_000_000.0] * len(BENCHMARK_CASES),
    )
    # Mark one case as incorrect on the leaderboard kernel.
    runtimes: list[float | None] = [500_000.0] * len(BENCHMARK_CASES)
    runtimes[0] = None
    bad = _make_scorecard(
        name="bad",
        leaderboard_id=LeaderboardKernelId.TTT,
        runtime_ns_per_case=runtimes,
    )
    cmp = _make_comparison(candidate=candidate, leaderboard=[bad])
    ranking = cmp.ranking_all_cases()
    assert [name for name, _ in ranking] == ["candidate"]


def test_ranking_nomask_true_keeps_kernel_incorrect_only_on_nomask_false() -> None:
    """A kernel incorrect on a nomask=False case is still ranked under nomask_true."""
    nomask_false_indices = [
        i for i, c in enumerate(BENCHMARK_CASES) if not c["nomask"]
    ]
    assert nomask_false_indices, "test fixture assumption: BENCHMARK_CASES has nomask=False entries"

    candidate = _make_scorecard(
        name="candidate",
        runtime_ns_per_case=[1_000_000.0] * len(BENCHMARK_CASES),
    )
    runtimes: list[float | None] = [500_000.0] * len(BENCHMARK_CASES)
    for i in nomask_false_indices:
        runtimes[i] = None  # incorrect on every nomask=False case
    ttt_like = _make_scorecard(
        name="ttt-like",
        leaderboard_id=LeaderboardKernelId.TTT,
        runtime_ns_per_case=runtimes,
    )
    cmp = _make_comparison(candidate=candidate, leaderboard=[ttt_like])

    # Excluded from all-cases.
    assert "ttt-like" not in dict(cmp.ranking_all_cases())
    # Kept in nomask=True ranking.
    nomask_ranking = dict(cmp.ranking_nomask_true())
    assert "ttt-like" in nomask_ranking
    assert math.isclose(nomask_ranking["ttt-like"], 500_000.0, rel_tol=1e-9)


def test_ranking_sorted_ascending_by_runtime() -> None:
    """Faster kernels rank first."""
    a = _make_scorecard(
        name="A",
        runtime_ns_per_case=[3_000_000.0] * len(BENCHMARK_CASES),
    )
    b = _make_scorecard(
        name="B",
        leaderboard_id=LeaderboardKernelId.SHIYEGAO,
        runtime_ns_per_case=[1_000_000.0] * len(BENCHMARK_CASES),
    )
    c = _make_scorecard(
        name="C",
        leaderboard_id=LeaderboardKernelId.WAQAR,
        runtime_ns_per_case=[2_000_000.0] * len(BENCHMARK_CASES),
    )
    cmp = _make_comparison(candidate=a, leaderboard=[b, c])
    ranking = cmp.ranking_all_cases()
    assert [name for name, _ in ranking] == ["B", "C", "A"]


def test_comparison_round_trips_through_json() -> None:
    candidate = _make_scorecard(
        name="candidate",
        runtime_ns_per_case=_all_correct_runtimes(),
    )
    leaderboard = [
        _make_scorecard(
            name="TTT",
            leaderboard_id=LeaderboardKernelId.TTT,
            published_geomean_us=2198.190,
            runtime_ns_per_case=_all_correct_runtimes(scale=2.0),
        )
    ]
    cmp = _make_comparison(candidate=candidate, leaderboard=leaderboard)
    blob = cmp.model_dump_json()
    restored = LeaderboardComparison.model_validate_json(blob)
    assert restored == cmp


def test_load_leaderboard_baseline_raises_when_missing(tmp_path, monkeypatch) -> None:
    """If no baselines exist for the requested GPU, raise FileNotFoundError with
    a hint to run bootstrap."""
    import gpu_forecasters.trimul.leaderboard_comparison as mod

    monkeypatch.setattr(mod, "LEADERBOARD_BASELINES_DIR", tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="bootstrap"):
        load_leaderboard_baseline(GpuKind.A100_80GB)


def _all_correct_runtimes(*, scale: float = 1.0) -> list[float | None]:
    """Distinct per-case runtimes so geomean is non-trivial. Returns
    list[float | None] for type-compatibility with _make_scorecard."""
    return [float((i + 1) * 1_000_000) * scale for i in range(len(BENCHMARK_CASES))]


_REF_SELF_CANDIDATE = """\
from gpu_forecasters.trimul.reference import ref_kernel

def custom_kernel(data):
    return ref_kernel(data)
"""


def _baselines_present(gpu: GpuKind) -> bool:
    return all(
        _baseline_path(gpu, kid).exists() for kid in LeaderboardKernelId
    )


@pytest.mark.modal
@pytest.mark.integration
@pytest.mark.skipif(
    not _baselines_present(GpuKind.A100_80GB),
    reason="A100-80GB baselines not bootstrapped; "
    "run `bootstrap-trimul-leaderboard --gpu A100-80GB`",
)
def test_compare_to_leaderboard_end_to_end_with_reference_kernel() -> None:
    """End-to-end smoke: scoring the reference kernel as the candidate produces
    a well-formed comparison, and the cached leaderboard scorecards match what's
    on disk (fixtures haven't drifted since bootstrap)."""
    cmp = compare_to_leaderboard(
        _REF_SELF_CANDIDATE,
        GpuKind.A100_80GB,
        candidate_name="ref",
    )
    assert cmp.candidate_scorecard.name == "ref"
    assert all(r.correct for r in cmp.candidate_scorecard.case_results)
    assert len(cmp.leaderboard_scorecards) == len(LeaderboardKernelId)
    # Reference kernel is the oracle, so every leaderboard kernel should beat
    # it on the nomask=True ranking (which all 5 are correct on).
    nomask_ranking = cmp.ranking_nomask_true()
    ref_pos = next(
        i for i, (name, _) in enumerate(nomask_ranking) if name == "ref"
    )
    assert ref_pos == len(nomask_ranking) - 1, (
        f"reference kernel should be slowest, got ranking={nomask_ranking}"
    )
