"""Integration test: v2 ``TriMulFeedbackMutationProvider`` exercised
through v3's ``MutationProvider`` protocol shape.

Per the round-1 handoff, the v2 mutation provider is supposed to
satisfy v3's protocol unchanged. This test pins that empirically:
fires one mutation against a real LLM, asserts a string comes back,
and that the string is plausibly Python (parseable as a module).

Also chains a real mutation into a surrogate forecast — the exact
data flow the v3 driver uses — to surface any failure mode that's
specific to LLM-mutated kernels rather than the seed kernel.

Run: ``uv run --env-file .env pytest -m integration tests/max_reward_puct/v3/integration/test_mutation_provider_live.py -v -s``
"""

from __future__ import annotations

import ast

import pytest

from gpu_forecasters.hill_climbing.domain import Evaluation
from gpu_forecasters.hill_climbing.scoring_providers.trimul import TriMulObservation
from gpu_forecasters.landscape_map.v2 import (
    HardwareContext,
    KernelImplementation,
    KernelRuntimeQuery,
    KernelTaskInfo,
)
from gpu_forecasters.landscape_map.v2.litellm_estimator import LlmSpeedupEstimator
from gpu_forecasters.max_reward_puct.v2.mutation_providers.trimul_feedback_mutation import (
    TriMulFeedbackMutationProvider,
)
from gpu_forecasters.max_reward_puct.v3.providers import MutationProvider
from gpu_forecasters.max_reward_puct.v3.scoring_providers import (
    CoroutineSpeedupEstimator,
)
from gpu_forecasters.trimul.core import InfrastructureFailureFeedback
from gpu_forecasters.trimul.seed_kernel import SEED_KERNEL_CODE


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


def _seed_evaluation() -> Evaluation[TriMulObservation]:
    """Synthetic 'infra-failure' evaluation. The mutation provider's
    prompt builder takes this branch when ``feedback`` is an
    InfrastructureFailureFeedback, which means it doesn't try to
    interpolate per-case timing (which we don't have here)."""
    return Evaluation[TriMulObservation](
        observation=TriMulObservation(
            feedback=InfrastructureFailureFeedback(reason="seed eval")
        ),
        reward=None,
    )


def test_v2_mutation_provider_satisfies_v3_protocol() -> None:
    """Type-level check: v2 provider conforms to v3's MutationProvider
    structurally. If this typechecks at import + runs without protocol
    error, the protocol seam works."""
    provider: MutationProvider[TriMulObservation] = TriMulFeedbackMutationProvider(
        model_slug="gemini/gemini-3-flash-preview",
        gpu_name="A100-80GB",
    )
    # Statically: the assignment above already proves protocol conformance
    # at type-check time. At runtime there is no isinstance check
    # (Protocol without runtime_checkable). We just confirm the object
    # has the required surface.
    assert hasattr(provider, "submit")
    assert hasattr(provider, "__enter__")
    assert hasattr(provider, "__exit__")


def test_one_mutation_returns_valid_python() -> None:
    """One real LLM mutation, assert the future resolves to parseable
    Python source. This is the per-call plumbing test."""
    provider = TriMulFeedbackMutationProvider(
        model_slug="gemini/gemini-3-flash-preview",
        gpu_name="A100-80GB",
    )
    with provider:
        future = provider.submit(SEED_KERNEL_CODE, _seed_evaluation())
        code = future.result(timeout=180.0)

    assert isinstance(code, str)
    assert len(code) > 100, f"suspiciously short mutation: {len(code)} chars"
    # Should be parseable as Python.
    ast.parse(code)
    print(f"\n  mutation length: {len(code)} chars")


def test_real_mutation_then_forecast_chain() -> None:
    """Smoke-equivalent chain: mutate the seed kernel, then forecast
    the speedup of the mutated candidate vs. the seed reference. This
    is the exact data shape the v3 driver feeds through the
    mutation→forecast phase. If the surrogate fails on real mutated
    kernel code (but passes on the seed kernel as candidate, per
    test_speedup_estimator_live.py), the bug is data-dependent on
    the mutation."""
    mutation_provider = TriMulFeedbackMutationProvider(
        model_slug="gemini/gemini-3-flash-preview",
        gpu_name="A100-80GB",
    )
    surrogate_inner = LlmSpeedupEstimator(
        model_slug="gemini/gemini-3-flash-preview",
        temperature=0.7,
        max_tokens=16384,
        request_timeout_s=180.0,
    )
    with mutation_provider:
        mutated_code = mutation_provider.submit(
            SEED_KERNEL_CODE, _seed_evaluation()
        ).result(timeout=180.0)

    print(f"\n  mutation length: {len(mutated_code)} chars")

    query = KernelRuntimeQuery(
        task=_TASK,
        reference=KernelImplementation(
            kernel_name="reference", code=SEED_KERNEL_CODE, runtime_ms=None
        ),
        candidate=KernelImplementation(
            kernel_name="candidate", code=mutated_code, runtime_ms=None
        ),
        hardware=_HARDWARE,
    )

    with CoroutineSpeedupEstimator(surrogate_inner) as adapter:
        future = adapter.submit(query)
        try:
            estimate, usage = future.result(timeout=240.0)
        except BaseException as exc:  # noqa: BLE001
            pytest.fail(
                f"surrogate failed on real mutated kernel "
                f"({len(mutated_code)} chars):\n"
                f"  {type(exc).__name__}: {exc!r}\n"
                f"  --- first 200 chars of mutation ---\n"
                f"  {mutated_code[:200]!r}"
            )

    total = sum(estimate.bin_probabilities.values())
    assert 0.99 <= total <= 1.01
    print(
        f"  predicted_bin={estimate.predicted_bin.name} "
        f"input_tokens={usage.input_tokens if usage else '?'} "
        f"output_tokens={usage.output_tokens if usage else '?'}"
    )
