"""Native v2 TriMul mutation provider.

Per-candidate contract: one ``submit(parent_code, evaluation)`` =
exactly one outbound ``litellm.acompletion(..., n=1)`` call producing
exactly one code string (or one failure). This is the shape the v2
event log atomically represents — it must match the shape of the
underlying work.

The provider owns a dedicated asyncio event loop on a background
thread. ``submit`` schedules a coroutine onto that loop via
``asyncio.run_coroutine_threadsafe`` and returns the resulting
``concurrent.futures.Future``. Concurrency is bounded by an
``asyncio.Semaphore`` owned by the loop.

Prompt formatting reuses the pure helpers from the v1 provider
module. Those helpers have no I/O — they format strings. Importing
the private helpers is a deliberate decision; duplicating the vendored
TriMul base prompt (~180 lines) would create a drift risk with zero
benefit.
"""

from __future__ import annotations

import asyncio
import threading
import traceback
from concurrent.futures import Future
from typing import Self

import litellm
from loguru import logger

from arid_badger.hill_climbing.domain import Evaluation
from arid_badger.hill_climbing.mutation_providers.trimul_feedback_mutation import (
    _build_base_prompt,
    _extract_last_python_codeblock,
    format_trimul_feedback_mutation_prompt,
)
from arid_badger.hill_climbing.scoring_providers.trimul import TriMulObservation
from arid_badger.trimul.core import InfrastructureFailureFeedback


class MutationError(RuntimeError):
    """Raised inside the provider's coroutine to signal a per-candidate
    failure. The v2 driver catches this and emits ``MutationFailed``."""


class TriMulFeedbackMutationProvider:
    """Per-candidate async TriMul mutation provider.

    Implements ``AsyncMutationProvider[TriMulObservation]``.
    """

    def __init__(
        self,
        *,
        model_slug: str,
        gpu_name: str,
        triton_version: str = "3.3.1",
        max_llm_concurrency: int = 8,
        num_retries: int = 4,
        request_timeout_s: float = 300.0,
        temperature: float = 1.0,
    ) -> None:
        if max_llm_concurrency < 1:
            raise ValueError("max_llm_concurrency must be >= 1")
        self._model_slug = model_slug
        self._base_prompt = _build_base_prompt(
            gpu_name=gpu_name, triton_version=triton_version
        )
        self._max_llm_concurrency = max_llm_concurrency
        self._num_retries = num_retries
        self._request_timeout_s = request_timeout_s
        self._temperature = temperature

        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._loop_ready = threading.Event()

    # --- Lifecycle ------------------------------------------------------

    def __enter__(self) -> Self:
        self._loop_thread = threading.Thread(
            target=self._run_loop, name="trimul-mutation-loop", daemon=True
        )
        self._loop_thread.start()
        # Block until the loop is running and the semaphore exists so
        # that a racing caller's ``submit`` sees a fully-initialized
        # provider.
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

    # --- Submit ---------------------------------------------------------

    def submit(
        self,
        parent_code: str,
        evaluation: Evaluation[TriMulObservation],
    ) -> Future[str]:
        if self._loop is None or self._semaphore is None:
            raise RuntimeError(
                "TriMulFeedbackMutationProvider must be entered as a "
                "context manager before submit()."
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
            return self._base_prompt
        return format_trimul_feedback_mutation_prompt(
            base_prompt=self._base_prompt,
            previous_kernel_code=parent_code,
            feedback=feedback,
        )

    async def _generate(self, prompt: str) -> str:
        assert self._semaphore is not None
        async with self._semaphore:
            try:
                response = await litellm.acompletion(
                    model=self._model_slug,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self._temperature,
                    num_retries=self._num_retries,
                    timeout=self._request_timeout_s,
                )
            except Exception as exc:
                logger.warning(
                    "TriMul mutation LLM call failed: {exc}\n{tb}",
                    exc=exc,
                    tb=traceback.format_exc(),
                )
                raise MutationError(f"litellm.acompletion failed: {exc}") from exc

            content = response.choices[0].message.content  # pyright: ignore[reportAttributeAccessIssue]
            if not content:
                raise MutationError("LLM returned empty content")
            code = _extract_last_python_codeblock(content)
            if not code:
                raise MutationError("no python code block extracted from response")
            return code
