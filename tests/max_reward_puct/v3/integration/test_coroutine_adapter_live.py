"""Integration test: ``CoroutineSpeedupEstimator`` over a real LLM
surrogate with concurrent submits.

The plain async surrogate test (``test_speedup_estimator_live.py``)
calls ``aestimate`` directly and passes. The v3 driver, in contrast,
goes through ``CoroutineSpeedupEstimator`` and fires multiple submits
concurrently. The first live-Modal smoke saw 2/2 forecasts fail in
exactly this configuration, so the failure must lie in the adapter or
under concurrent submission. This test pins which.

Run: ``uv run --env-file .env pytest -m integration tests/max_reward_puct/v3/integration/test_coroutine_adapter_live.py -v -s``
"""

from __future__ import annotations

import pytest

from arid_badger.landscape_map.v2 import (
    HardwareContext,
    KernelImplementation,
    KernelRuntimeQuery,
    KernelTaskInfo,
)
from arid_badger.landscape_map.v2.litellm_estimator import LlmSpeedupEstimator
from arid_badger.max_reward_puct.v3.scoring_providers import (
    CoroutineSpeedupEstimator,
)
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


def _query() -> KernelRuntimeQuery:
    return KernelRuntimeQuery(
        task=_TASK,
        reference=KernelImplementation(
            kernel_name="reference", code=SEED_KERNEL_CODE, runtime_ms=None
        ),
        candidate=KernelImplementation(
            kernel_name="candidate", code=SEED_KERNEL_CODE, runtime_ms=None
        ),
        hardware=_HARDWARE,
    )


def test_single_submit_through_adapter() -> None:
    """Sanity: one forecast through the adapter should match the
    direct-async test (which passes)."""
    inner = LlmSpeedupEstimator(
        model_slug="gemini/gemini-3-flash-preview",
        temperature=0.7,
        max_tokens=16384,
        request_timeout_s=180.0,
    )
    with CoroutineSpeedupEstimator(inner) as adapter:
        future = adapter.submit(_query())
        estimate, _usage = future.result(timeout=180.0)
    total = sum(estimate.bin_probabilities.values())
    assert 0.99 <= total <= 1.01


def test_two_concurrent_submits_through_adapter() -> None:
    """The exact configuration the v3 driver uses on the first step
    with samples_per_parent=2: two forecast submits in flight at the
    same time. If both fail here, the adapter or concurrency is the
    bug — not the surrogate or the driver loop."""
    inner = LlmSpeedupEstimator(
        model_slug="gemini/gemini-3-flash-preview",
        temperature=0.7,
        max_tokens=16384,
        request_timeout_s=180.0,
    )
    errors: list[BaseException] = []
    estimates_count = 0
    with CoroutineSpeedupEstimator(inner) as adapter:
        f1 = adapter.submit(_query())
        f2 = adapter.submit(_query())
        for fut in (f1, f2):
            try:
                estimate, _usage = fut.result(timeout=180.0)
                total = sum(estimate.bin_probabilities.values())
                assert 0.99 <= total <= 1.01
                estimates_count += 1
            except BaseException as exc:  # noqa: BLE001 — we want every error type
                errors.append(exc)

    if errors:
        # Surface every error so we know whether it's a single recurring
        # mode (e.g. all "model did not call any tool") or different.
        msg = "\n".join(f"  {type(e).__name__}: {e!r}" for e in errors)
        pytest.fail(
            f"{len(errors)}/2 concurrent submits failed (succeeded={estimates_count}):\n{msg}"
        )
