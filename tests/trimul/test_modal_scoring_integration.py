"""Integration tests for Modal-based TriMul scoring (requires Modal + GPU).

Gated behind the ``modal`` and ``integration`` markers — not collected by
default. Run with:

    uv run --env-file .env pytest -m "modal and integration" \\
        tests/trimul/test_modal_scoring_integration.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_forecasters.trimul.cases import BENCHMARK_CASES, CORRECTNESS_CASES
from gpu_forecasters.trimul.leaderboard_comparison import LEADERBOARD_REGISTRY
from gpu_forecasters.trimul.modal_scoring import modal_trimul_scoring_session
from gpu_forecasters.typing_utils import is_ok


_FIXTURES = Path(__file__).parent / "fixtures"
_LEADERBOARD = _FIXTURES / "leaderboard"


# A100 leaderboard ranking (problem 496) at the time the kernels were
# scraped. Runtimes are the site-reported geometric-mean across 7 benchmark
# cases, in microseconds. Order is load-bearing for the ranking test.
# Sourced from the canonical registry in gpu_forecasters.trimul.leaderboard_comparison.
_LEADERBOARD_KERNELS: list[tuple[str, str, float]] = [
    (entry.fixture_filename, entry.display_name, entry.published_geomean_us)
    for entry in LEADERBOARD_REGISTRY.values()
]


pytestmark = [pytest.mark.modal, pytest.mark.integration]


# Known-bad (user, case_idx) pairs: the kernel is wrong on this specific
# public test case but still ranks on the leaderboard because popcorn grades
# on secret shapes. Evidence: the other 4 top-5 kernels all pass every case,
# so this is kernel-specific drift on small masked shapes, not oracle drift.
#   TTT case 1: {seqlen:32, bs:1, dim:128, nomask:False, seed:1092} — 91% of
#     output elements mismatch, ~0.06–0.14 abs diff. Systematic, not numerical.
#   TTT case 3: {seqlen:64, bs:2, dim:256, nomask:False, seed:210284} — same
#     pattern, 1.8M mismatched elements.
#   TTT case 6: {seqlen:256, bs:1, dim:128, nomask:False, seed:10432} — 6.7M
#     mismatched elements.
#   TTT case 8: {seqlen:1024, bs:1, dim:384, nomask:False, seed:53121} — 220M
#     mismatched elements.
#   TTT case 10: {seqlen:1024, bs:1, dim:768, nomask:False, seed:4921} — 346M
#     mismatched elements.
#   TTT case 16: {seqlen:1024, bs:1, dim:384, nomask:False, cauchy} — 217M
#     mismatched elements.
#   TTT case 17: {seqlen:1024, bs:1, dim:768, nomask:False, cauchy} — 343M
#     mismatched elements.
# Pattern: every nomask=False case fails with ~91% element mismatch at
# ~0.06-0.14 abs diff. nomask=True cases (0, 2, 4, 5, 7, 9, 11-15) all pass.
# Kernel has a systematic masked-path bug; still ranks on leaderboard because
# popcorn grades on secret shapes. Other 4 top-5 kernels pass all 18 cases.
_LEADERBOARD_KNOWN_BAD: set[tuple[str, int]] = {
    ("TTT", 1), ("TTT", 3), ("TTT", 6), ("TTT", 8),
    ("TTT", 10), ("TTT", 16), ("TTT", 17),
}


_REF_SELF_CANDIDATE = """\
from gpu_forecasters.trimul.reference import ref_kernel

def custom_kernel(data):
    return ref_kernel(data)
"""


_ZEROS_CANDIDATE = """\
import torch

def custom_kernel(data):
    input_tensor = data[0]
    return torch.zeros_like(input_tensor)
"""


_RUNTIME_ERR_CANDIDATE = """\
def custom_kernel(data):
    raise RuntimeError("boom")
"""


_SYNTAX_ERR_CANDIDATE = "def custom_kernel(data:\n    return data\n"


def _load_fixture(name: str) -> str:
    return (_FIXTURES / name).read_text()


def test_reference_self_consistency_all_cases() -> None:
    """Scoring the reference kernel against itself must yield correct=True
    on every correctness case. Proves the vendored oracle chain
    (verbose_allclose + DisableCuDNNTF32 + ref_kernel) is internally consistent."""
    with modal_trimul_scoring_session() as score:
        for idx, case in enumerate(CORRECTNESS_CASES):
            results = score(_REF_SELF_CANDIDATE, [case])
            assert len(results) == 1
            result = results[0]
            assert is_ok(result), f"case {idx}: infrastructure failure {result}"
            exec_result = result.unwrap()
            assert exec_result.correct is True, (
                f"case {idx} ({case}) failed: {exec_result.error_message}"
            )


def test_identity_starter_kit() -> None:
    src = _load_fixture("reference_submission.py")
    with modal_trimul_scoring_session() as score:
        results = score(src, [BENCHMARK_CASES[0]])
    assert len(results) == 1
    result = results[0]
    assert is_ok(result)
    exec_result = result.unwrap()
    assert exec_result.correct is True
    assert exec_result.runtime_ns > 0
    assert exec_result.ref_runtime_ns > 0
    ratio = exec_result.runtime_ns / exec_result.ref_runtime_ns
    assert 0.5 < ratio < 2.0, f"starter kit ratio out of range: {ratio}"


def test_wrong_output() -> None:
    with modal_trimul_scoring_session() as score:
        results = score(_ZEROS_CANDIDATE, [BENCHMARK_CASES[0]])
    assert len(results) == 1
    result = results[0]
    assert is_ok(result)
    exec_result = result.unwrap()
    assert exec_result.correct is False
    assert exec_result.failure_kind == "incorrect"
    assert exec_result.error_message


def test_runtime_exception() -> None:
    with modal_trimul_scoring_session() as score:
        results = score(_RUNTIME_ERR_CANDIDATE, [BENCHMARK_CASES[0]])
    assert len(results) == 1
    result = results[0]
    assert is_ok(result)
    exec_result = result.unwrap()
    assert exec_result.correct is False
    assert exec_result.failure_kind == "runtime_error"
    assert "boom" in exec_result.runtime_error


def test_syntax_error() -> None:
    with modal_trimul_scoring_session() as score:
        results = score(_SYNTAX_ERR_CANDIDATE, [BENCHMARK_CASES[0]])
    assert len(results) == 1
    result = results[0]
    assert is_ok(result)
    exec_result = result.unwrap()
    assert exec_result.correct is False
    assert exec_result.failure_kind == "compile_failed"
    assert exec_result.compilation_error


@pytest.mark.parametrize(
    "filename,user",
    [(name, user) for name, user, _ in _LEADERBOARD_KERNELS],
    ids=[user for _, user, _ in _LEADERBOARD_KERNELS],
)
def test_leaderboard_kernel_correct_on_all_cases(filename: str, user: str) -> None:
    """A top-5 A100 leaderboard submission must pass correctness on every
    CORRECTNESS_CASES entry. These kernels were graded by popcorn against
    the same reference implementation, so any mismatch points at drift in
    our port of the oracle / candidate resolver, not at the kernels."""
    source = (_LEADERBOARD / filename).read_text()
    with modal_trimul_scoring_session() as score:
        for idx, case in enumerate(CORRECTNESS_CASES):
            if (user, idx) in _LEADERBOARD_KNOWN_BAD:
                continue
            results = score(source, [case])
            assert len(results) == 1
            result = results[0]
            assert is_ok(result), f"{user} case {idx}: infra failure {result}"
            exec_result = result.unwrap()
            assert exec_result.correct is True, (
                f"{user} failed correctness on case {idx} ({case}): "
                f"{exec_result.error_message[:500]}"
            )


def test_leaderboard_kernels_ranking_geomean_nomask_true_benchmarks() -> None:
    """Port-correctness check: geometric-mean runtime ranking of the top-5
    A100 leaderboard kernels must match their reported order.

    Popcorn scores the leaderboard as geo-mean across all 7 BENCHMARK_CASES
    (reported μs: TTT 2198, shiyegao 2273, emmet 2370, arseni 4532, waqar
    4919 — clear ~2x separation between top-3 and bottom-2, well outside
    timing noise on this hardware).

    Why we restrict to nomask=True cases here: the #1 kernel (TTT) has a bug
    on nomask=False inputs that makes it incorrect on BENCHMARK_CASES[2] and
    [5] (and on every nomask=False CORRECTNESS_CASES entry — see
    _LEADERBOARD_KNOWN_BAD above for the full story). Other 4 top-5 kernels
    are correct on all 7 cases. To keep the comparison apples-to-apples, we
    geo-mean over the 5 nomask=True benchmark cases (indices 0, 1, 3, 4, 6)
    for every kernel. This is not literally what popcorn measures, but the
    ranking signal is preserved: the top-3 still dominate the bottom-2 by
    ~2x on pure dense shapes, and TTT stays in the comparison rather than
    being dropped for a kernel-specific bug unrelated to our port.

    Asserts max(top3_geomean) < min(bot2_geomean). Does NOT assert intra-top-3
    order — the 2.2/2.3/2.4ms gap is within single-case timing noise.

    Each kernel scores all 5 nomask=True benchmark cases in a single Modal
    container call. 5 kernels = 5 container calls, parallelised via thread pool."""
    import math
    from concurrent.futures import ThreadPoolExecutor

    ranking_cases = [case for case in BENCHMARK_CASES if case["nomask"]]
    assert len(ranking_cases) == 5, (
        f"expected 5 nomask=True benchmark cases, got {len(ranking_cases)}"
    )

    sources = {
        name: (_LEADERBOARD / name).read_text()
        for name, _, _ in _LEADERBOARD_KERNELS
    }

    with modal_trimul_scoring_session() as score:
        def score_kernel(name: str) -> tuple[str, float]:
            results = score(sources[name], ranking_cases)
            runtimes: list[float] = []
            for i, result in enumerate(results):
                assert is_ok(result), f"{name} case {i}: infra failure"
                exec_result = result.unwrap()
                assert exec_result.correct is True, (
                    f"{name} case {i}: incorrect — "
                    f"{exec_result.error_message[:200]}"
                )
                runtimes.append(exec_result.runtime_ns)
            geomean = math.exp(sum(math.log(r) for r in runtimes) / len(runtimes))
            return name, geomean

        kernel_names = [name for name, _, _ in _LEADERBOARD_KERNELS]
        with ThreadPoolExecutor(max_workers=len(kernel_names)) as executor:
            results_list = list(executor.map(score_kernel, kernel_names))

    geomeans = dict(results_list)

    top3_names = [n for n, _, _ in _LEADERBOARD_KERNELS[:3]]
    bot2_names = [n for n, _, _ in _LEADERBOARD_KERNELS[3:]]
    top3 = [geomeans[n] for n in top3_names]
    bot2 = [geomeans[n] for n in bot2_names]
    assert max(top3) < min(bot2), (
        f"top-3 did not all beat bottom-2 on geo-mean across nomask=True "
        f"benchmark cases: top3={dict(zip(top3_names, top3))}, "
        f"bot2={dict(zip(bot2_names, bot2))}"
    )
