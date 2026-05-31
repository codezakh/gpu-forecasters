"""Bin-balanced ordered-pair sampler.

Given a pool of candidates split across problems, the sampler enumerates
ordered (anchor, candidate) pairs within each problem, buckets them by
the bin of their pairwise speedup, and yields at most ``target_per_bin``
pairs from each (problem, bin) cell, water-filled across problems so
each problem with supply contributes its share before any one problem
doubles up.

Output is a deterministic function of the inputs and ``rng_seed``: same
candidates + same target + same seed → same selection. Tests can pin
all three.

The sampler is a pure function (modulo the RNG seed) — no I/O. The
experiment that calls it owns persistence.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict

from pydantic import BaseModel

from gpu_forecasters.landscape_map.v2.domain import SpeedupBin

from .domain import CandidateKernel, LabeledPair, ProblemId


# Bin index range for sampling balance. SpeedupBin uses 1..8 for
# successes; bin 0 (failure) does not arise from pairwise ratios over
# already-correct candidates, so we ignore it.
_BINS_FOR_BALANCE: tuple[int, ...] = tuple(range(1, 9))


def _bin_of_log2(log2_speedup: float) -> int:
    return int(SpeedupBin.from_speedup(2.0 ** log2_speedup))


class PerProblemSupply(BaseModel, frozen=True):
    """Diagnostic: how many ordered pairs exist per (problem, bin)
    *before* sampling, for one side."""

    problem_id: ProblemId
    n_candidates: int
    pairs_per_bin: dict[int, int]


class SidedSampleReport(BaseModel, frozen=True):
    """Diagnostic for one source's contribution to the final dataset."""

    n_pairs_picked: int
    target_per_bin: int
    picked_per_bin: dict[int, int]
    supply_per_bin: dict[int, int]
    shortfall_per_bin: dict[int, int]
    distinct_kernels_touched: int
    per_problem_picks: dict[ProblemId, int]
    per_problem_supply: list[PerProblemSupply]


class BalancedPairSamplerConfig(BaseModel, frozen=True):
    """Knobs for one call to :func:`build_balanced_pair_dataset`.

    ``target_per_bin`` is the only scaling axis. With perfect supply
    each side yields ``8 * target_per_bin`` pairs; if any (problem-side,
    bin) cell is supply-limited the realized total is lower.

    ``rng_seed`` controls which specific pairs are chosen within an
    over-supplied (problem, bin) cell — it does NOT affect the totals
    or the bin balance.
    """

    target_per_bin: int
    rng_seed: int = 0


class BalancedPairDataset(BaseModel, frozen=True):
    """Materialized dataset returned by the sampler.

    ``pairs`` is the union across all named sources, in source-emit
    order. ``per_source_reports`` carries diagnostics per source so
    callers can inspect the bin balance and per-problem distribution
    without re-deriving it.
    """

    config: BalancedPairSamplerConfig
    pairs: list[LabeledPair]
    per_source_reports: dict[str, SidedSampleReport]


def _enumerate_per_problem_pair_supply(
    candidates: list[CandidateKernel],
) -> dict[ProblemId, dict[int, list[tuple[int, int, float]]]]:
    """problem_id -> bin -> list of (anchor_idx, cand_idx, log2_ratio).

    Indices refer to position within that problem's candidate list (held
    by the caller, in :func:`_sample_one_source`).
    """
    by_problem: dict[ProblemId, list[CandidateKernel]] = defaultdict(list)
    for c in candidates:
        by_problem[c.problem_id].append(c)

    out: dict[ProblemId, dict[int, list[tuple[int, int, float]]]] = {}
    for prob, cands in by_problem.items():
        per_bin: dict[int, list[tuple[int, int, float]]] = defaultdict(list)
        n = len(cands)
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                ratio = cands[i].runtime / cands[j].runtime
                if ratio <= 0:
                    continue
                log2_ratio = math.log2(ratio)
                b = _bin_of_log2(log2_ratio)
                per_bin[b].append((i, j, log2_ratio))
        out[prob] = dict(per_bin)
    return out


def _water_fill_one_bin(
    problems_with_supply: list[tuple[ProblemId, list[tuple[int, int, float]]]],
    target: int,
    rng: random.Random,
) -> list[tuple[ProblemId, int, int, float]]:
    """Distribute ``target`` picks across problems uniformly via
    rounds. Each round, each problem contributes up to ``share`` picks
    where ``share = ceil(remaining / n_problems)``, capped by remaining
    supply.

    Returns list of (problem_id, anchor_idx, cand_idx, log2_ratio).
    """
    # Shuffle each problem's pool deterministically against the same RNG.
    shuffled: list[tuple[ProblemId, list[tuple[int, int, float]]]] = []
    for prob, lst in problems_with_supply:
        copy = list(lst)
        rng.shuffle(copy)
        shuffled.append((prob, copy))

    picks: list[tuple[ProblemId, int, int, float]] = []
    remaining = target
    pools = shuffled
    while remaining > 0 and pools:
        n_p = len(pools)
        # Per-problem share for this round, at least 1 so we always
        # make progress.
        share = max(1, remaining // n_p)
        next_pools: list[tuple[ProblemId, list[tuple[int, int, float]]]] = []
        for prob, lst in pools:
            take = min(share, len(lst), remaining)
            for _ in range(take):
                a, c, r = lst.pop()
                picks.append((prob, a, c, r))
                remaining -= 1
                if remaining == 0:
                    break
            if lst:
                next_pools.append((prob, lst))
            if remaining == 0:
                break
        pools = next_pools
    return picks


def _sample_one_source(
    candidates: list[CandidateKernel],
    config: BalancedPairSamplerConfig,
) -> tuple[list[LabeledPair], SidedSampleReport]:
    """Sample one source's candidates into a per-bin-balanced list of
    ordered pairs."""
    by_problem: dict[ProblemId, list[CandidateKernel]] = defaultdict(list)
    for c in candidates:
        by_problem[c.problem_id].append(c)

    supply = _enumerate_per_problem_pair_supply(candidates)

    rng = random.Random(config.rng_seed)
    picked_pairs: list[LabeledPair] = []
    picked_per_bin: Counter[int] = Counter()
    supply_per_bin: Counter[int] = Counter()
    per_problem_picks: Counter[ProblemId] = Counter()
    distinct_touched: set[tuple[ProblemId, str]] = set()

    # Aggregate per-bin supply for diagnostics.
    for prob, by_bin in supply.items():
        for b, lst in by_bin.items():
            supply_per_bin[b] += len(lst)

    for b in _BINS_FOR_BALANCE:
        problems_with_supply = [
            (prob, by_bin.get(b, [])) for prob, by_bin in supply.items()
        ]
        problems_with_supply = [
            (prob, lst) for prob, lst in problems_with_supply if lst
        ]
        if not problems_with_supply:
            continue
        chosen = _water_fill_one_bin(
            problems_with_supply, target=config.target_per_bin, rng=rng
        )
        for prob, a_idx, c_idx, log2_ratio in chosen:
            cands = by_problem[prob]
            anchor = cands[a_idx]
            candidate = cands[c_idx]
            picked_pairs.append(
                LabeledPair(
                    problem_id=prob,
                    anchor_code_hash=anchor.code_hash,
                    anchor_code=anchor.code,
                    candidate_code_hash=candidate.code_hash,
                    candidate_code=candidate.code,
                    log2_speedup=log2_ratio,
                )
            )
            picked_per_bin[b] += 1
            per_problem_picks[prob] += 1
            distinct_touched.add((prob, anchor.code_hash))
            distinct_touched.add((prob, candidate.code_hash))

    shortfall = {
        b: max(0, config.target_per_bin - picked_per_bin.get(b, 0))
        for b in _BINS_FOR_BALANCE
    }

    per_problem_supply = [
        PerProblemSupply(
            problem_id=prob,
            n_candidates=len(by_problem[prob]),
            pairs_per_bin={b: len(lst) for b, lst in by_bin.items()},
        )
        for prob, by_bin in supply.items()
    ]

    report = SidedSampleReport(
        n_pairs_picked=len(picked_pairs),
        target_per_bin=config.target_per_bin,
        picked_per_bin=dict(picked_per_bin),
        supply_per_bin=dict(supply_per_bin),
        shortfall_per_bin=shortfall,
        distinct_kernels_touched=len(distinct_touched),
        per_problem_picks=dict(per_problem_picks),
        per_problem_supply=per_problem_supply,
    )
    return picked_pairs, report


def build_balanced_pair_dataset(
    sources: dict[str, list[CandidateKernel]],
    config: BalancedPairSamplerConfig,
) -> BalancedPairDataset:
    """Sample a balanced pair dataset across multiple sources.

    Each source contributes ``target_per_bin`` pairs per bin
    independently — this is what enforces the equal source split. The
    final dataset is the union, in source-emit order, plus per-source
    diagnostic reports.

    Sources are passed as a dict so the caller picks the source labels
    that appear in the report; the sampler itself doesn't bake in any
    source taxonomy.
    """
    if not sources:
        raise ValueError("sources must contain at least one named source")
    all_pairs: list[LabeledPair] = []
    reports: dict[str, SidedSampleReport] = {}
    for label, candidates in sources.items():
        pairs, report = _sample_one_source(candidates, config)
        all_pairs.extend(pairs)
        reports[label] = report
    return BalancedPairDataset(
        config=config, pairs=all_pairs, per_source_reports=reports
    )
