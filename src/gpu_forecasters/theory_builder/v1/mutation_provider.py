"""Hypothesis-conditioned TriMul mutation provider.

Wraps the existing v2 ``TriMulFeedbackMutationProvider`` interface — it
emits one ``submit(parent_code, evaluation)`` per candidate as a
``Future[str]``. The hypothesis is injected as additional guidance
into the prompt; the rest of the prompt (base TriMul context +
feedback summary) is reused verbatim.

The hypothesis is fixed for the lifetime of the provider — i.e. for
the duration of one inner search. The driver constructs a fresh
provider for each outer-loop step.
"""

from __future__ import annotations

import asyncio
import threading
import traceback
from concurrent.futures import Future
from typing import Self

import litellm
from loguru import logger

from gpu_forecasters.hill_climbing.domain import Evaluation
from gpu_forecasters.hill_climbing.mutation_providers.trimul_feedback_mutation import (
    _build_base_prompt,
    _extract_last_python_codeblock,
    format_trimul_feedback_mutation_prompt,
)
from gpu_forecasters.hill_climbing.scoring_providers.trimul import TriMulObservation
from gpu_forecasters.theory_builder.v1.domain import Hypothesis
from gpu_forecasters.trimul.core import InfrastructureFailureFeedback


class MutationError(RuntimeError):
    """Per-candidate failure signalled to the v2 driver, which emits
    ``MutationFailed``."""


def _format_hypothesis_block(hypothesis: Hypothesis) -> str:
    refs = (
        "\n".join(f"  - {r}" for r in hypothesis.code_references)
        if hypothesis.code_references
        else "  (none)"
    )
    return f"""\
=== Theory-builder hypothesis ===

Before continuing, here is the current hypothesis your inner search is
testing. Treat this as a strong prior on what change to attempt; you
are NOT required to follow it exactly, but the outer loop is going to
read your generated kernels and assess whether the hypothesis was
right. Concrete, falsifiable interventions are more useful than
generic optimizations.

Bottleneck claim:
{hypothesis.bottleneck}

Intervention proposed:
{hypothesis.intervention}

Prediction:
{hypothesis.prediction}

Code references:
{refs}

================================
"""


class HypothesisConditionedTriMulMutationProvider:
    """Per-candidate async TriMul mutation provider, conditioned on
    one ``Hypothesis``.

    Implements ``AsyncMutationProvider[TriMulObservation]``.
    """

    def __init__(
        self,
        *,
        hypothesis: Hypothesis,
        model_slug: str,
        gpu_name: str,
        triton_version: str = "3.3.1",
        max_llm_concurrency: int = 8,
        num_retries: int = 4,
        request_timeout_s: float = 300.0,
        temperature: float = 1.0,
        max_tokens: int | None = None,
    ) -> None:
        if max_llm_concurrency < 1:
            raise ValueError("max_llm_concurrency must be >= 1")
        if max_tokens is not None and max_tokens < 1:
            raise ValueError("max_tokens must be >= 1 when set")
        self._hypothesis = hypothesis
        self._model_slug = model_slug
        self._base_prompt = _build_base_prompt(
            gpu_name=gpu_name, triton_version=triton_version
        )
        self._hypothesis_block = _format_hypothesis_block(hypothesis)
        self._max_llm_concurrency = max_llm_concurrency
        self._num_retries = num_retries
        self._request_timeout_s = request_timeout_s
        self._temperature = temperature
        self._max_tokens = max_tokens

        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._loop_ready = threading.Event()

    # --- Lifecycle ----------------------------------------------------

    def __enter__(self) -> Self:
        self._loop_thread = threading.Thread(
            target=self._run_loop,
            name="theory-mutation-loop",
            daemon=True,
        )
        self._loop_thread.start()
        self._loop_ready.wait()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        if self._loop is not None and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._loop_thread is not None:
            self._loop_thread.join(timeout=10.0)
        self._loop = None
        self._loop_thread = None
        self._semaphore = None
        self._loop_ready.clear()

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._loop = loop
        self._semaphore = asyncio.Semaphore(self._max_llm_concurrency)
        self._loop_ready.set()
        try:
            loop.run_forever()
        finally:
            loop.close()

    # --- Submit -------------------------------------------------------

    def submit(
        self,
        parent_code: str,
        evaluation: Evaluation[TriMulObservation],
    ) -> Future[str]:
        if self._loop is None or self._semaphore is None:
            raise RuntimeError(
                "HypothesisConditionedTriMulMutationProvider must be entered "
                "as a context manager before submit()."
            )
        prompt = self._build_prompt(parent_code, evaluation)
        coro = self._generate(prompt)
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def _build_prompt(
        self,
        parent_code: str,
        evaluation: Evaluation[TriMulObservation],
    ) -> str:
        feedback = evaluation.observation.feedback
        if isinstance(feedback, InfrastructureFailureFeedback):
            base = self._base_prompt
        else:
            base = format_trimul_feedback_mutation_prompt(
                base_prompt=self._base_prompt,
                previous_kernel_code=parent_code,
                feedback=feedback,
            )
        # Hypothesis block goes after the feedback block — it's the
        # most recent, most action-relevant context. Same place
        # callers normally drop additional steering instructions.
        return base.rstrip() + "\n\n" + self._hypothesis_block

    async def _generate(self, prompt: str) -> str:
        assert self._semaphore is not None
        async with self._semaphore:
            try:
                if self._max_tokens is None:
                    response = await litellm.acompletion(
                        model=self._model_slug,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self._temperature,
                        num_retries=self._num_retries,
                        timeout=self._request_timeout_s,
                    )
                else:
                    response = await litellm.acompletion(
                        model=self._model_slug,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self._temperature,
                        num_retries=self._num_retries,
                        timeout=self._request_timeout_s,
                        max_tokens=self._max_tokens,
                    )
            except Exception as exc:
                logger.warning(
                    "Hypothesis-mutation LLM call failed: {exc}\n{tb}",
                    exc=exc,
                    tb=traceback.format_exc(),
                )
                raise MutationError(
                    f"litellm.acompletion failed: {exc}"
                ) from exc

            content = response.choices[0].message.content  # pyright: ignore[reportAttributeAccessIssue]
            if not content:
                raise MutationError("LLM returned empty content")
            code = _extract_last_python_codeblock(content)
            if not code:
                raise MutationError(
                    "no python code block extracted from response"
                )
            return code


__all__ = [
    "HypothesisConditionedTriMulMutationProvider",
    "MutationError",
]
