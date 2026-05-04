"""Tests for the v1 bin-balanced pair sampler."""

from __future__ import annotations

from collections import Counter

from arid_badger.datasets.balanced_pair_sampling.v1 import (
    BalancedPairSamplerConfig,
    CandidateKernel,
    LabeledPair,
    ProblemId,
    build_balanced_pair_dataset,
)
from arid_badger.landscape_map.v2.domain import SpeedupBin


def _candidate(prob: str, code: str, runtime: float) -> CandidateKernel:
    return CandidateKernel(
        problem_id=ProblemId(prob),
        code_hash=f"h_{code}",
        code=code,
        runtime=runtime,
    )


def _bin_of_pair(p: LabeledPair) -> int:
    return int(SpeedupBin.from_speedup(2.0 ** p.log2_speedup))


def test_strict_balance_when_supply_is_plentiful() -> None:
    # Half-octave spacing (2^0.25 step) so pair ratios land in every
    # bin including the odd-half-octave ones (bins 3, 4, 5, 7).
    cands_a = [_candidate("A", f"a{i}", 2.0 ** (i * 0.25 - 2)) for i in range(16)]
    cands_b = [_candidate("B", f"b{i}", 2.0 ** (i * 0.25 - 2)) for i in range(16)]
    config = BalancedPairSamplerConfig(target_per_bin=4, rng_seed=0)
    result = build_balanced_pair_dataset(
        sources={"alpha": cands_a, "beta": cands_b}, config=config
    )

    # Two sources, target 4 per bin per source, 8 bins -> 64 total
    assert len(result.pairs) == 64
    bin_counts = Counter(_bin_of_pair(p) for p in result.pairs)
    assert all(bin_counts[b] == 8 for b in range(1, 9))  # 4 per source x 2

    for label in ["alpha", "beta"]:
        report = result.per_source_reports[label]
        assert report.n_pairs_picked == 32
        assert all(report.picked_per_bin[b] == 4 for b in range(1, 9))
        assert all(s == 0 for s in report.shortfall_per_bin.values())


def test_shortfall_recorded_when_bin_supply_caps() -> None:
    # Two candidates with runtimes 1.0 and 2.0 yield only 2 ordered pairs:
    # (1,2) -> ratio 0.5 -> log2 = -1 -> bin 2
    # (2,1) -> ratio 2.0 -> log2 = +1 -> bin 6
    cands = [
        _candidate("only", "x", 1.0),
        _candidate("only", "y", 2.0),
    ]
    config = BalancedPairSamplerConfig(target_per_bin=10, rng_seed=0)
    result = build_balanced_pair_dataset(sources={"s": cands}, config=config)

    assert len(result.pairs) == 2
    bin_counts = Counter(_bin_of_pair(p) for p in result.pairs)
    assert bin_counts[2] == 1
    assert bin_counts[6] == 1

    report = result.per_source_reports["s"]
    assert report.n_pairs_picked == 2
    # Shortfall reported for every bin we couldn't fill to 10.
    for b in range(1, 9):
        if b in (2, 6):
            assert report.shortfall_per_bin[b] == 9
        else:
            assert report.shortfall_per_bin[b] == 10


def test_water_fills_evenly_across_problems() -> None:
    # Two problems with identical wide-spread runtime sets. Pair supply is
    # symmetric across both problems and across bins. Water-filling at
    # target_per_bin=8 should split picks 4/4 per problem in each bin.
    runtimes = [2.0 ** (i * 0.25 - 2) for i in range(16)]
    cands_a = [_candidate("A", f"a{i}", rt) for i, rt in enumerate(runtimes)]
    cands_b = [_candidate("B", f"b{i}", rt) for i, rt in enumerate(runtimes)]
    config = BalancedPairSamplerConfig(target_per_bin=8, rng_seed=0)
    result = build_balanced_pair_dataset(
        sources={"src": cands_a + cands_b}, config=config
    )

    # Per-bin split between A and B should be even.
    by_bin_problem: dict[int, Counter[ProblemId]] = {}
    for p in result.pairs:
        b = _bin_of_pair(p)
        by_bin_problem.setdefault(b, Counter())[p.problem_id] += 1
    for b, counts in by_bin_problem.items():
        # target=8 with 2 supplying problems -> 4 each.
        assert counts[ProblemId("A")] == counts[ProblemId("B")] == 4, (
            f"bin {b} not split evenly: {dict(counts)}"
        )


def test_deterministic_under_same_seed() -> None:
    cands = [_candidate("p", f"c{i}", float(i + 1)) for i in range(10)]
    config = BalancedPairSamplerConfig(target_per_bin=3, rng_seed=42)
    r1 = build_balanced_pair_dataset(sources={"s": cands}, config=config)
    r2 = build_balanced_pair_dataset(sources={"s": cands}, config=config)
    assert [p.model_dump() for p in r1.pairs] == [p.model_dump() for p in r2.pairs]


def test_log2_speedup_matches_runtime_ratio() -> None:
    cands = [
        _candidate("p", "fast", 1.0),
        _candidate("p", "slow", 4.0),
    ]
    config = BalancedPairSamplerConfig(target_per_bin=10, rng_seed=0)
    result = build_balanced_pair_dataset(sources={"s": cands}, config=config)
    # Pair anchor=fast (rt=1) candidate=slow (rt=4) -> ratio = 1/4 -> log2 = -2
    # Pair anchor=slow (rt=4) candidate=fast (rt=1) -> ratio = 4 -> log2 = +2
    log2s = sorted(p.log2_speedup for p in result.pairs)
    assert log2s == [-2.0, 2.0]


def test_zero_or_one_candidate_problem_yields_no_pairs() -> None:
    # Problem A has only one candidate -> no pairs possible.
    # Problem B has plenty.
    cands = [
        _candidate("A", "lonely", 1.0),
        _candidate("B", "b0", 1.0),
        _candidate("B", "b1", 2.0),
    ]
    config = BalancedPairSamplerConfig(target_per_bin=10, rng_seed=0)
    result = build_balanced_pair_dataset(sources={"s": cands}, config=config)
    assert all(p.problem_id == ProblemId("B") for p in result.pairs)
