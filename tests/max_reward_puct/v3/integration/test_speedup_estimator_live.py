"""Integration test: ``LlmSpeedupEstimator`` against a real TriMul pair.

Fires one surrogate forecast against the actual kernel-runtime query
shape v3 sends in production — i.e. the seed kernel as both reference
and candidate. Surfaces the real exception (parse error, timeout,
provider error) directly, so we don't have to discover failure modes
through the full v3 driver round-trip.

This is the test that surfaced the round-1 surrogate failure mode the
top-level smoke caught indirectly. Keep it as the first stop when the
surrogate misbehaves.

Run: ``uv run --env-file .env pytest -m integration tests/max_reward_puct/v3/integration/test_speedup_estimator_live.py -v -s``
"""

from __future__ import annotations

import asyncio

import pytest

from arid_badger.landscape_map.v2 import (
    HardwareContext,
    KernelImplementation,
    KernelRuntimeQuery,
    KernelTaskInfo,
)
from arid_badger.landscape_map.v2.litellm_estimator import LlmSpeedupEstimator
from arid_badger.trimul.seed_kernel import SEED_KERNEL_CODE


pytestmark = pytest.mark.integration


_HARDWARE = HardwareContext(
    device_name="NVIDIA A100-SXM4-80GB",
    compute_capability=(8, 0),
    total_global_memory_gb=80.0,
    multiprocessor_count=108,
    max_threads_per_multiprocessor=2048,
    clock_rate_ghz=1.41,
    memory_clock_rate_ghz=1.512,
    memory_bus_width_bits=5120,
)
_TASK = KernelTaskInfo(op_name="trimul", level_id=0, task_id=0)


def _query(candidate_code: str) -> KernelRuntimeQuery:
    return KernelRuntimeQuery(
        task=_TASK,
        reference=KernelImplementation(
            kernel_name="reference", code=SEED_KERNEL_CODE, runtime_ms=None
        ),
        candidate=KernelImplementation(
            kernel_name="candidate", code=candidate_code, runtime_ms=None
        ),
        hardware=_HARDWARE,
    )


@pytest.mark.parametrize(
    ("model_slug", "max_tokens"),
    [
        ("gemini/gemini-3-flash-preview", 4096),
        ("gemini/gemini-3-flash-preview", 16384),
    ],
    ids=["gemini3-flash-4k", "gemini3-flash-16k"],
)
def test_estimator_returns_valid_forecast_on_real_trimul(
    model_slug: str, max_tokens: int
) -> None:
    """Single forecast on a TriMul ref/candidate pair (both = seed).

    Asserts the estimator returns a well-formed forecast. Failure modes
    we want to surface as the test's exception, not as a swallowed
    ``ForecastFailed`` in the driver:
    - parse error ("model did not call any tool") — common when a
      thinking model exhausts its token budget on reasoning
    - litellm timeout / provider error
    - tool-args validation error
    """
    estimator = LlmSpeedupEstimator(
        model_slug=model_slug,
        temperature=0.7,
        max_tokens=max_tokens,
        request_timeout_s=180.0,
    )
    estimate, usage = asyncio.run(estimator.aestimate(_query(SEED_KERNEL_CODE)))

    # Distribution well-formed.
    total = sum(estimate.bin_probabilities.values())
    assert 0.99 <= total <= 1.01, (
        f"bin probabilities should sum to ~1.0; got {total}, "
        f"raw_probability_sum={estimate.raw_probability_sum}"
    )
    # The argmax bin should be the predicted bin.
    argmax_bin = max(
        estimate.bin_probabilities, key=lambda b: estimate.bin_probabilities[b]
    )
    assert argmax_bin == estimate.predicted_bin, (
        f"predicted_bin={estimate.predicted_bin.name} but argmax="
        f"{argmax_bin.name}"
    )
    # Print usage for visibility — surrogate cost matters at scale.
    if usage is not None:
        print(
            f"\n  model={model_slug} max_tokens={max_tokens} "
            f"input_tokens={usage.input_tokens} output_tokens={usage.output_tokens} "
            f"predicted_bin={estimate.predicted_bin.name}"
        )
