"""Integration tests for split CPU-compile / GPU-benchmark Modal scoring.

Mirror `test_modal_scoring.py`. Require Modal auth; skipped by default —
opt in via `uv run --env-file .env pytest -m "integration and modal" \\
    tests/test_modal_split_scoring.py -v`.
"""

import time

import pytest

from arid_badger.kernelbench.modal_split_scoring import modal_split_scoring_session
from arid_badger.typing_utils import is_err, is_ok

from tests.test_modal_scoring import (
    BROKEN_KERNEL_CODE,
    CORRECT_KERNEL_CODE,
    REFERENCE_KERNEL_CODE,
)


@pytest.mark.integration
@pytest.mark.modal
@pytest.mark.slow
def test_modal_split_scoring_correct_kernel() -> None:
    with modal_split_scoring_session(
        gpu="L4", num_correct_trials=1, num_perf_trials=5
    ) as score:
        result = score(CORRECT_KERNEL_CODE, REFERENCE_KERNEL_CODE)

    assert is_ok(result), f"Expected Ok, got Err: {result}"
    exec_result = result.unwrap()
    assert exec_result.compiled, f"Kernel failed to compile: {exec_result.metadata}"
    assert exec_result.correctness, f"Kernel produced incorrect output: {exec_result.metadata}"
    assert exec_result.runtime > 0, f"Expected positive runtime, got {exec_result.runtime}"
    assert exec_result.ref_runtime > 0, (
        f"Expected positive ref_runtime, got {exec_result.ref_runtime}"
    )


@pytest.mark.integration
@pytest.mark.modal
@pytest.mark.slow
def test_modal_split_scoring_broken_kernel() -> None:
    with modal_split_scoring_session(
        gpu="L4", num_correct_trials=1, num_perf_trials=5
    ) as score:
        result = score(BROKEN_KERNEL_CODE, REFERENCE_KERNEL_CODE)

    if is_ok(result):
        exec_result = result.unwrap()
        assert not exec_result.correctness or not exec_result.compiled, (
            "Expected broken kernel to fail, but got a successful result"
        )
    else:
        assert is_err(result)


@pytest.mark.integration
@pytest.mark.modal
@pytest.mark.slow
def test_modal_split_scoring_cache_hit() -> None:
    """Second call on the same kernel should short-circuit nvcc via the volume cache."""
    with modal_split_scoring_session(
        gpu="L4", num_correct_trials=1, num_perf_trials=5
    ) as score:
        t0 = time.perf_counter()
        cold = score(CORRECT_KERNEL_CODE, REFERENCE_KERNEL_CODE)
        t_cold = time.perf_counter() - t0

        t0 = time.perf_counter()
        warm = score(CORRECT_KERNEL_CODE, REFERENCE_KERNEL_CODE)
        t_warm = time.perf_counter() - t0

    assert is_ok(cold), f"cold call failed: {cold}"
    assert is_ok(warm), f"warm call failed: {warm}"
    assert cold.unwrap().compiled and cold.unwrap().correctness
    assert warm.unwrap().compiled and warm.unwrap().correctness
    assert t_warm < t_cold, (
        f"cache hit should be faster than cold compile: cold={t_cold:.1f}s warm={t_warm:.1f}s"
    )
