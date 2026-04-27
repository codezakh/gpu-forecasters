"""Lifecycle and integration tests for the v2 Gemini causal conv1d
mutation provider.

Unit-level tests: ``__enter__``/``__exit__`` start/stop the loop thread
cleanly; ``submit`` without ``__enter__`` raises a clear error.

Integration tests (marked, opt-in): hit the real Gemini API for one
candidate to confirm end-to-end plumbing. Mirror of
``test_trimul_mutation_provider``.
"""

from __future__ import annotations

import pytest

from arid_badger.causal_conv1d.core import InfrastructureFailureFeedback
from arid_badger.hill_climbing.domain import Evaluation
from arid_badger.hill_climbing.scoring_providers.causal_conv1d import (
    CausalConv1dObservation,
)
from arid_badger.max_reward_puct.v2.mutation_providers.causal_conv1d_feedback_mutation import (
    CausalConv1dFeedbackMutationProvider,
)


def _seed_eval() -> Evaluation[CausalConv1dObservation]:
    return Evaluation[CausalConv1dObservation](
        observation=CausalConv1dObservation(
            feedback=InfrastructureFailureFeedback(reason="bootstrap"),
            per_case_results=[],
        ),
        reward=None,
    )


def test_submit_without_enter_raises() -> None:
    p = CausalConv1dFeedbackMutationProvider(
        model_slug="gemini/gemini-2.5-flash", gpu_name="A100-80GB"
    )
    with pytest.raises(RuntimeError, match="context manager"):
        _ = p.submit("# placeholder", _seed_eval())


def test_lifecycle_start_and_stop() -> None:
    p = CausalConv1dFeedbackMutationProvider(
        model_slug="gemini/gemini-2.5-flash", gpu_name="A100-80GB"
    )
    with p:
        assert p._loop is not None  # pyright: ignore[reportPrivateUsage]
        assert p._loop.is_running()  # pyright: ignore[reportPrivateUsage]
    # After exit, the loop should be torn down.
    assert p._loop is None  # pyright: ignore[reportPrivateUsage]


@pytest.mark.integration
def test_real_gemini_call_returns_code() -> None:
    """Live Gemini call. Asserts that submit returns a Future that
    eventually resolves to a string with reasonable Triton-shaped
    content."""
    p = CausalConv1dFeedbackMutationProvider(
        model_slug="gemini/gemini-2.5-flash",
        gpu_name="A100-80GB",
        max_llm_concurrency=2,
        request_timeout_s=300.0,
    )
    seed_code = "# placeholder seed kernel\n"
    with p:
        fut = p.submit(seed_code, _seed_eval())
        code = fut.result(timeout=600)
    assert isinstance(code, str)
    assert len(code) > 100, "expected a non-trivial code response"
    assert "def custom_kernel" in code or "custom_kernel" in code
