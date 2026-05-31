"""V2 async mutation provider for KernelBench problems.

Implements ``AsyncMutationProvider[KernelBenchObservation]``. Per-candidate
contract: one ``submit(parent_code, evaluation)`` issues exactly one outbound
``litellm.acompletion(..., n=1)`` and produces exactly one code string (or one
``MutationError`` for the v2 driver to log as ``MutationFailed``). Matches the
v2 event log's atomic shape — the unit of work and the unit of logging agree.

The provider owns a single asyncio event loop on a dedicated background
thread. ``submit`` schedules a coroutine onto that loop via
``asyncio.run_coroutine_threadsafe`` and returns a ``concurrent.futures.Future``
to the v2 driver. Concurrency is bounded by an ``asyncio.Semaphore`` owned by
the loop; ``max_llm_concurrency`` is the cap on simultaneous outbound LLM
calls.

The mutation prompt is rendered from a Jinja template at
``gpu_forecasters.kernelbench.v2.providers.prompts.mutation``. The template
carries no problem-specific examples — it gives general optimization
guidance and states the structural contract (the ``ModelNew`` +
``load_inline`` skeleton) that the eval pipeline requires. A short
skeleton example is embedded inline below as a structural template, not a
problem-relevant demonstration.

On infrastructure failure (no useful prior signal), the prompt is rendered
without the ``Previous attempt`` block — the LLM sees the task fresh.
"""

from __future__ import annotations

import asyncio
import threading
import time
import traceback
from concurrent.futures import Future
from datetime import datetime, timezone
from typing import Any, Literal, Self

import litellm
from jinja2 import Environment, PackageLoader, StrictUndefined
from loguru import logger
from pydantic import BaseModel

from gpu_forecasters.code_extraction import extract_last_python_codeblock
from gpu_forecasters.hill_climbing.domain import Evaluation
from gpu_forecasters.hill_climbing.scoring_providers.kernelbench import (
    KernelBenchObservation,
)
from gpu_forecasters.invocation_sink import InvocationSink, code_sha256
from gpu_forecasters.kernelbench.core import (
    InfrastructureFailureFeedback,
    KernelExecutionFeedback,
)
from gpu_forecasters.max_reward_puct.v2.providers import AsyncMutationProvider
from gpu_forecasters.typing_utils import implements


class MutationError(RuntimeError):
    """Raised inside the provider's coroutine to signal a per-candidate
    failure. The v2 driver catches this and emits ``MutationFailed``."""


class KernelBenchMutationRecord(BaseModel, frozen=True):
    """Invocation record for a single v2 KernelBench mutation call.

    One record per ``submit(...)`` — matches the v2 atomic unit. On
    failure (``MutationError``), ``child_code_sha256`` is None and
    ``failure_reason`` carries the message.
    """

    kind: Literal["kernelbench_mutation_v2"] = "kernelbench_mutation_v2"
    parent_code_sha256: str
    child_code_sha256: str | None
    model_slug: str
    input_tokens: int | None
    output_tokens: int | None
    wall_clock_seconds: float
    failure_reason: str | None
    timestamp_utc: str


# ---------------------------------------------------------------------------
# Truncation budgets + helpers
# ---------------------------------------------------------------------------

# Tracebacks tail-truncate (deepest frame is at the bottom — usually the
# actionable signal); everything else head-truncates.
_MAX_COMPILATION_ERROR_CHARS = 2000
_MAX_RUNTIME_ERROR_CHARS = 1000
_MAX_TRACEBACK_CHARS = 3000
_MAX_INCORRECT_ERROR_CHARS = 2000


def _truncate_head(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n[truncated]"


def _truncate_tail(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return f"[truncated]\n...{text[-max_chars:]}"


# ---------------------------------------------------------------------------
# Skeleton example — structural template, intentionally trivial. Shows the
# minimal load_inline + ModelNew shape the harness expects. Not a hint about
# any specific problem.
# ---------------------------------------------------------------------------

_SKELETON_EXAMPLE = '''\
import torch
import torch.nn as nn
from torch.utils.cpp_extension import load_inline

_CUDA_SOURCE = r"""
#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void scaled_relu_kernel(const float* __restrict__ x,
                                   float scale,
                                   float* __restrict__ out,
                                   int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v = x[i] * scale;
        out[i] = v > 0.f ? v : 0.f;
    }
}

torch::Tensor scaled_relu_cuda(torch::Tensor x, double scale) {
    auto out = torch::empty_like(x);
    int n = x.numel();
    int block = 256;
    int grid = (n + block - 1) / block;
    scaled_relu_kernel<<<grid, block>>>(x.data_ptr<float>(),
                                        static_cast<float>(scale),
                                        out.data_ptr<float>(),
                                        n);
    return out;
}
"""

_CPP_SOURCE = "torch::Tensor scaled_relu_cuda(torch::Tensor x, double scale);"

_ext = load_inline(
    name="scaled_relu_ext",
    cpp_sources=_CPP_SOURCE,
    cuda_sources=_CUDA_SOURCE,
    functions=["scaled_relu_cuda"],
    extra_cuda_cflags=["-O3"],
    verbose=False,
)


class ModelNew(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.contiguous()
        return _ext.scaled_relu_cuda(x, self.scale)
'''


# ---------------------------------------------------------------------------
# Jinja environment
# ---------------------------------------------------------------------------

_JINJA_ENV = Environment(
    loader=PackageLoader("gpu_forecasters.kernelbench.v2.providers", "prompts"),
    autoescape=False,
    undefined=StrictUndefined,
    keep_trailing_newline=True,
)
_MUTATION_TEMPLATE = _JINJA_ENV.get_template("mutation.j2")


def render_mutation_prompt(
    *,
    reference_kernel_code: str,
    previous_kernel_code: str | None,
    feedback: KernelExecutionFeedback | None,
) -> str:
    """Render the mutation prompt for a given parent state.

    ``feedback`` is ``None`` when the prior evaluation was an infrastructure
    failure (or, conceptually, when there is no usable prior signal); the
    template then omits the ``Previous attempt`` section and the LLM sees
    the task fresh. ``previous_kernel_code`` is required iff ``feedback``
    is not None — a feedback section without the code it refers to would
    be incoherent.
    """
    if feedback is not None and previous_kernel_code is None:
        raise ValueError(
            "previous_kernel_code is required when feedback is provided"
        )

    context: dict[str, Any] = {
        "reference_module": reference_kernel_code.rstrip(),
        "feedback": feedback,
        "previous_kernel_code": (
            previous_kernel_code.rstrip() if previous_kernel_code else None
        ),
        "skeleton_example": _SKELETON_EXAMPLE.rstrip(),
        # Pre-truncated payloads. The template references these by the
        # ``_truncated`` suffix so the truncation policy lives in one place
        # (here) rather than being duplicated across template branches.
        "compilation_error_truncated": "",
        "runtime_error_truncated": "",
        "runtime_error_traceback_truncated": "",
        "correctness_issue_truncated": "",
    }

    if feedback is not None:
        match feedback.kind:
            case "compile_failed":
                context["compilation_error_truncated"] = _truncate_head(
                    feedback.compilation_error, _MAX_COMPILATION_ERROR_CHARS  # pyright: ignore[reportAttributeAccessIssue]
                )
            case "runtime_error":
                context["runtime_error_truncated"] = _truncate_head(
                    feedback.runtime_error, _MAX_RUNTIME_ERROR_CHARS  # pyright: ignore[reportAttributeAccessIssue]
                )
                context["runtime_error_traceback_truncated"] = _truncate_tail(
                    feedback.runtime_error_traceback, _MAX_TRACEBACK_CHARS  # pyright: ignore[reportAttributeAccessIssue]
                )
            case "incorrect":
                context["correctness_issue_truncated"] = _truncate_head(
                    feedback.correctness_issue, _MAX_INCORRECT_ERROR_CHARS  # pyright: ignore[reportAttributeAccessIssue]
                )
            case "success":
                # SuccessFeedback fields go straight to the template.
                pass

    return _MUTATION_TEMPLATE.render(**context)


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class KernelBenchFeedbackMutationProvider:
    """Per-candidate async mutation provider for KernelBench problems.

    Implements ``AsyncMutationProvider[KernelBenchObservation]``.

    ``max_tokens`` is the per-call output-token cap. ``None`` (the default)
    leaves litellm's behaviour untouched — appropriate for Gemini models,
    where an explicit cap historically interacts poorly with the thinking
    budget. Set explicitly (e.g. 32000) for Together-hosted gpt-oss, where
    the provider's default ~4-8K cap truncates mid-kernel.
    """

    def __init__(
        self,
        *,
        reference_kernel_code: str,
        model_slug: str,
        max_llm_concurrency: int = 8,
        num_retries: int = 4,
        request_timeout_s: float = 600.0,
        temperature: float = 1.0,
        max_tokens: int | None = None,
        invocation_sink: InvocationSink | None = None,
    ) -> None:
        if max_llm_concurrency < 1:
            raise ValueError("max_llm_concurrency must be >= 1")
        if max_tokens is not None and max_tokens < 1:
            raise ValueError("max_tokens must be >= 1 when set")
        if not reference_kernel_code:
            raise ValueError("reference_kernel_code must be non-empty")

        self._reference_kernel_code = reference_kernel_code
        self._model_slug = model_slug
        self._max_llm_concurrency = max_llm_concurrency
        self._num_retries = num_retries
        self._request_timeout_s = request_timeout_s
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._invocation_sink = invocation_sink

        self._loop: asyncio.AbstractEventLoop | None = None
        self._loop_thread: threading.Thread | None = None
        self._semaphore: asyncio.Semaphore | None = None
        self._loop_ready = threading.Event()

    # --- Lifecycle ------------------------------------------------------

    def __enter__(self) -> Self:
        self._loop_thread = threading.Thread(
            target=self._run_loop,
            name="kernelbench-mutation-loop",
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
        # The semaphore must be constructed on the loop that will acquire
        # it — asyncio synchronization primitives bind to the running loop
        # on first await.
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
        evaluation: Evaluation[KernelBenchObservation],
    ) -> Future[str]:
        if self._loop is None or self._semaphore is None:
            raise RuntimeError(
                f"{type(self).__name__} must be entered as a context manager "
                "before submit()."
            )
        prompt = self._build_prompt(parent_code, evaluation)
        coro = self._generate(prompt, parent_code=parent_code)
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def _build_prompt(
        self,
        parent_code: str,
        evaluation: Evaluation[KernelBenchObservation],
    ) -> str:
        feedback = evaluation.observation.feedback
        if isinstance(feedback, InfrastructureFailureFeedback):
            # No usable prior signal. Render without the ``Previous attempt``
            # section so the LLM does not condition on noise.
            return render_mutation_prompt(
                reference_kernel_code=self._reference_kernel_code,
                previous_kernel_code=None,
                feedback=None,
            )
        return render_mutation_prompt(
            reference_kernel_code=self._reference_kernel_code,
            previous_kernel_code=parent_code,
            feedback=feedback,
        )

    async def _generate(self, prompt: str, *, parent_code: str) -> str:
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

        parent_sha = code_sha256(parent_code)
        start_time_s = time.perf_counter()

        async with self._semaphore:
            try:
                response = await litellm.acompletion(**kwargs)
            except Exception as exc:
                logger.warning(
                    "KernelBench mutation LLM call failed: {exc}\n{tb}",
                    exc=exc,
                    tb=traceback.format_exc(),
                )
                reason = f"litellm.acompletion failed: {exc}"
                self._record(
                    parent_sha=parent_sha,
                    child_sha=None,
                    input_tokens=None,
                    output_tokens=None,
                    start_time_s=start_time_s,
                    failure_reason=reason,
                )
                raise MutationError(reason) from exc

            usage = response.usage  # pyright: ignore[reportAttributeAccessIssue]
            input_tokens = usage.prompt_tokens if usage is not None else None
            output_tokens = usage.completion_tokens if usage is not None else None

            content = response.choices[0].message.content  # pyright: ignore[reportAttributeAccessIssue]
            if not content:
                self._record(
                    parent_sha=parent_sha,
                    child_sha=None,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    start_time_s=start_time_s,
                    failure_reason="LLM returned empty content",
                )
                raise MutationError("LLM returned empty content")
            code = extract_last_python_codeblock(content)
            if not code:
                self._record(
                    parent_sha=parent_sha,
                    child_sha=None,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    start_time_s=start_time_s,
                    failure_reason="no python code block extracted from response",
                )
                raise MutationError("no python code block extracted from response")
            self._record(
                parent_sha=parent_sha,
                child_sha=code_sha256(code),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                start_time_s=start_time_s,
                failure_reason=None,
            )
            return code

    # --- Sink bookkeeping ---------------------------------------------

    def _record(
        self,
        *,
        parent_sha: str,
        child_sha: str | None,
        input_tokens: int | None,
        output_tokens: int | None,
        start_time_s: float,
        failure_reason: str | None,
    ) -> None:
        if self._invocation_sink is None:
            return
        self._invocation_sink.record(
            KernelBenchMutationRecord(
                parent_code_sha256=parent_sha,
                child_code_sha256=child_sha,
                model_slug=self._model_slug,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                wall_clock_seconds=time.perf_counter() - start_time_s,
                failure_reason=failure_reason,
                timestamp_utc=datetime.now(timezone.utc).isoformat(),
            )
        )


implements(AsyncMutationProvider[KernelBenchObservation])(
    KernelBenchFeedbackMutationProvider
)


__all__ = [
    "KernelBenchFeedbackMutationProvider",
    "KernelBenchMutationRecord",
    "MutationError",
    "render_mutation_prompt",
    "extract_last_python_codeblock",
]
