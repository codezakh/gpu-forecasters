"""Async mutation provider for the compound-observation search loop.

Implements ``AsyncMutationProvider[CompoundObservation[CaseSpeedupT]]``.
Each ``submit(parent_code, evaluation)`` produces exactly one mutated
kernel from the LLM. The prompt is dispatched on the parent's
observation arm:

* ``RealObservation`` whose inner feedback is one of the four in-band
  ``KernelExecutionFeedback`` arms → real-eval feedback prompt.
* ``RealObservation`` whose inner feedback is
  ``InfrastructureFailureFeedback`` → zero-shot base prompt (no
  candidate-side signal worth feeding back).
* ``ForecastObservation`` → forecast feedback prompt.

This provider is intentionally standalone — it does not wrap or
delegate to ``GpuModeKernelFeedbackMutationProvider``. Lifecycle,
asyncio loop, semaphore, LiteLLM call, and prompt rendering are all
internal so the compound module can evolve without coordinating.
"""

from __future__ import annotations

import asyncio
import threading
import traceback
from concurrent.futures import Future
from typing import Any, Generic, Self

import litellm
from loguru import logger

from gpu_forecasters.abstaining_evaluation.v1.observation import (
    CompoundObservation,
    ForecastObservation,
    RealObservation,
)
from gpu_forecasters.abstaining_evaluation.v1.prompts import (
    build_base_prompt,
    format_forecast_feedback_prompt,
    format_real_eval_feedback_prompt,
)
from gpu_forecasters.code_extraction import extract_last_python_codeblock
from gpu_forecasters.gpu_mode_kernel.core import (
    CaseSpeedupT,
    GpuModeKernelObservation,
    InfrastructureFailureFeedback,
)
from gpu_forecasters.gpu_mode_kernel.kernel_pack import KernelPack, TestArgsT
from gpu_forecasters.hill_climbing.domain import Evaluation


class CompoundMutationError(RuntimeError):
    """Raised inside the provider's coroutine to signal a per-candidate
    failure. The v2 driver catches this and emits ``MutationFailed``."""


class CompoundFeedbackMutationProvider(Generic[TestArgsT, CaseSpeedupT]):
    """Per-candidate async mutation provider for the compound search loop.

    Implements
    ``AsyncMutationProvider[CompoundObservation[CaseSpeedupT]]``.

    Construction is intentionally close to
    ``GpuModeKernelFeedbackMutationProvider``'s shape so wiring at the
    experiment level is symmetric — the only structural difference is
    that this provider's ``submit`` accepts an ``Evaluation`` whose
    observation may be either arm of ``CompoundObservation``.
    """

    def __init__(
        self,
        *,
        pack: KernelPack[TestArgsT, CaseSpeedupT],
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
        self._pack = pack
        self._model_slug = model_slug
        self._base_prompt = build_base_prompt(
            pack, gpu_name=gpu_name, triton_version=triton_version
        )
        self._max_llm_concurrency = max_llm_concurrency
        self._num_retries = num_retries
        self._request_timeout_s = request_timeout_s
        self._temperature = temperature
        self._max_tokens = max_tokens

        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._loop_ready = threading.Event()

    # --- Lifecycle ------------------------------------------------------

    def __enter__(self) -> Self:
        self._loop_thread = threading.Thread(
            target=self._run_loop,
            name=f"{self._pack.name}-compound-mutation-loop",
            daemon=True,
        )
        self._loop_thread.start()
        _ = self._loop_ready.wait()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        if self._loop is not None and self._loop.is_running():
            _ = self._loop.call_soon_threadsafe(self._loop.stop)
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
        evaluation: Evaluation[CompoundObservation[CaseSpeedupT]],
    ) -> Future[str]:
        if self._loop is None or self._semaphore is None:
            raise RuntimeError(
                f"{type(self).__name__} must be entered as a context manager "
                "before submit()."
            )
        prompt = self._build_prompt(parent_code, evaluation)
        coro = self._generate(prompt)
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def _build_prompt(
        self,
        parent_code: str,
        evaluation: Evaluation[CompoundObservation[CaseSpeedupT]],
    ) -> str:
        observation = evaluation.observation
        match observation:
            case ForecastObservation() as forecast:
                return format_forecast_feedback_prompt(
                    base_prompt=self._base_prompt,
                    parent_code=parent_code,
                    forecast=forecast,
                )
            case RealObservation(inner=inner):
                return self._build_real_eval_prompt(parent_code, inner)

    def _build_real_eval_prompt(
        self,
        parent_code: str,
        inner: GpuModeKernelObservation[CaseSpeedupT],
    ) -> str:
        if isinstance(inner.feedback, InfrastructureFailureFeedback):
            # No in-band signal worth feeding back; zero-shot.
            return self._base_prompt
        return format_real_eval_feedback_prompt(
            base_prompt=self._base_prompt,
            parent_code=parent_code,
            feedback=inner.feedback,
        )

    async def _generate(self, prompt: str) -> str:
        assert self._semaphore is not None
        kwargs: dict[str, Any] = {
            "model": self._model_slug,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._temperature,
            "num_retries": self._num_retries,
            "timeout": self._request_timeout_s,
        }
        if self._max_tokens is not None:
            kwargs["max_tokens"] = self._max_tokens
        async with self._semaphore:
            try:
                response = await litellm.acompletion(**kwargs)
            except Exception as exc:
                logger.warning(
                    "{name} compound mutation LLM call failed: {exc}\n{tb}",
                    name=self._pack.name,
                    exc=exc,
                    tb=traceback.format_exc(),
                )
                raise CompoundMutationError(
                    f"litellm.acompletion failed: {exc}"
                ) from exc

            content = response.choices[0].message.content  # pyright: ignore[reportAttributeAccessIssue]
            if not content:
                raise CompoundMutationError("LLM returned empty content")
            code = extract_last_python_codeblock(content)
            if not code:
                raise CompoundMutationError(
                    "no python code block extracted from response"
                )
            return code


__all__ = [
    "CompoundFeedbackMutationProvider",
    "CompoundMutationError",
]
