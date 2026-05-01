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
``arid_badger.kernelbench.providers.prompts.mutation``. The template carries
no problem-specific examples — it gives general optimization guidance and
states the structural contract (the ``ModelNew`` + ``load_inline`` skeleton)
that the eval pipeline requires. A short skeleton example is embedded inline
below as a structural template, not a problem-relevant demonstration.

On infrastructure failure (no useful prior signal), the prompt is rendered
without the ``Previous attempt`` block — the LLM sees the task fresh.
"""

from __future__ import annotations

import asyncio
import re
import threading
import traceback
from concurrent.futures import Future
from typing import Any, Self

import litellm
from jinja2 import Environment, PackageLoader, StrictUndefined
from loguru import logger

from arid_badger.hill_climbing.domain import Evaluation
from arid_badger.hill_climbing.scoring_providers.kernelbench import (
    KernelBenchObservation,
)
from arid_badger.kernelbench.core import (
    InfrastructureFailureFeedback,
    KernelExecutionFeedback,
)
from arid_badger.max_reward_puct.v2.providers import AsyncMutationProvider
from arid_badger.typing_utils import implements


class MutationError(RuntimeError):
    """Raised inside the provider's coroutine to signal a per-candidate
    failure. The v2 driver catches this and emits ``MutationFailed``."""


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

# Lifts the regex from ``arid_badger.gpu_mode_kernel.prompts.extract_last_python_codeblock``.
# Picks the LAST python block — the prompt instructs the model to put its
# final code in one trailing block, but reasoning models often emit drafts
# above that final block.

_PYTHON_CODEBLOCK_RE = re.compile(
    r"```python\n(?!```)(.*?)(?:\n```)?(?=\n```|$)",
    re.DOTALL,
)


def extract_last_python_codeblock(text: str) -> str | None:
    matches = list(_PYTHON_CODEBLOCK_RE.finditer(text))
    if not matches:
        return None
    code = matches[-1].group(1).rstrip()
    return code or None


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
    loader=PackageLoader("arid_badger.kernelbench.providers", "prompts"),
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
        coro = self._generate(prompt)
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
                    "KernelBench mutation LLM call failed: {exc}\n{tb}",
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


implements(AsyncMutationProvider[KernelBenchObservation])(
    KernelBenchFeedbackMutationProvider
)


__all__ = [
    "KernelBenchFeedbackMutationProvider",
    "MutationError",
    "render_mutation_prompt",
    "extract_last_python_codeblock",
]
