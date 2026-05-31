"""Generic v2 mutation provider for gpu-mode-style kernels.

Generalizes ``gpu_forecasters.max_reward_puct.v2.mutation_providers.{trimul,causal_conv1d}_feedback_mutation``:
asyncio loop thread, semaphore-bounded fan-out, lifecycle, and code
extraction lift verbatim. The kernel-specific surfaces collapse onto
the ``KernelPack`` (``kernel_description_body`` and
``case_speedup_format``).

Per-candidate contract: one ``submit(parent_code, evaluation)`` =
exactly one outbound ``litellm.acompletion(..., n=1)`` producing
exactly one code string (or one ``MutationError``). Matches the v2
event log's atomic shape.
"""

from __future__ import annotations

import asyncio
import threading
import traceback
from concurrent.futures import Future
from typing import Any, Generic, Self

import litellm
from loguru import logger

from gpu_forecasters.gpu_mode_kernel.core import (
    CaseSpeedupT,
    GpuModeKernelObservation,
    InfrastructureFailureFeedback,
)
from gpu_forecasters.gpu_mode_kernel.kernel_pack import KernelPack, TestArgsT
from gpu_forecasters.gpu_mode_kernel.prompts import (
    build_base_prompt,
    extract_last_python_codeblock,
    format_feedback_prompt,
)
from gpu_forecasters.hill_climbing.domain import Evaluation


class MutationError(RuntimeError):
    """Raised inside the provider's coroutine to signal a per-candidate
    failure. The v2 driver catches this and emits ``MutationFailed``."""


class GpuModeKernelFeedbackMutationProvider(Generic[TestArgsT, CaseSpeedupT]):
    """Per-candidate async mutation provider for gpu-mode-style kernels.

    Implements ``AsyncMutationProvider[GpuModeKernelObservation[CaseSpeedupT]]``.

    ``max_tokens=None`` is the right setting for Gemini 3 Flash (which
    rejects an explicit cap); Together gpt-oss requires an explicit
    32k cap.
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
            name=f"{self._pack.name}-mutation-loop",
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

    # --- Submit ---------------------------------------------------------

    def submit(
        self,
        parent_code: str,
        evaluation: Evaluation[GpuModeKernelObservation[CaseSpeedupT]],
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
        evaluation: Evaluation[GpuModeKernelObservation[CaseSpeedupT]],
    ) -> str:
        feedback = evaluation.observation.feedback
        if isinstance(feedback, InfrastructureFailureFeedback):
            return self._base_prompt
        return format_feedback_prompt(
            base_prompt=self._base_prompt,
            previous_kernel_code=parent_code,
            feedback=feedback,
        )

    async def _generate(self, prompt: str) -> str:
        assert self._semaphore is not None
        # ``max_tokens`` is conditional because Gemini 3 Flash rejects
        # an explicit cap, while Together gpt-oss requires one.
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
                    "{name} mutation LLM call failed: {exc}\n{tb}",
                    name=self._pack.name,
                    exc=exc,
                    tb=traceback.format_exc(),
                )
                raise MutationError(f"litellm.acompletion failed: {exc}") from exc

            content = response.choices[0].message.content  # pyright: ignore[reportAttributeAccessIssue]
            if not content:
                raise MutationError("LLM returned empty content")
            code = extract_last_python_codeblock(content)
            if not code:
                raise MutationError("no python code block extracted from response")
            return code
