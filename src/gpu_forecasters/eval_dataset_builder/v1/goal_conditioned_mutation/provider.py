"""Goal-conditioned kernel mutation provider — pack-generic.

A near-duplicate of ``GpuModeKernelFeedbackMutationProvider`` (in
``arid_badger.gpu_mode_kernel.providers.v2_feedback_mutation``) whose
prompt is rendered from a Jinja template that frames the task as
"land in a target speedup band" rather than "maximize speed."

The duplication is deliberate: the two prompts serve different
purposes (max-reward search vs. target-band search) and we want each
to be edit-able as a single-file diff. Lifecycle (asyncio loop, semaphore,
context-manager pair) and the litellm submit/extract path are
structurally identical to the upstream provider — copying them
verbatim avoids subtle drift from the reference implementation.

The ``render_prompt`` and ``build_render_context`` helpers are
module-level and pure; tests assert on their output without standing
up the asyncio loop or making LLM calls.
"""

from __future__ import annotations

import asyncio
import math
import threading
import traceback
from concurrent.futures import Future
from typing import Any, Generic, Self

import litellm
from jinja2 import Environment, PackageLoader, StrictUndefined, select_autoescape
from loguru import logger

from arid_badger.gpu_mode_kernel.core import (
    CaseSpeedupT,
    CompileFailedFeedback,
    GpuModeKernelObservation,
    IncorrectFeedback,
    InfrastructureFailureFeedback,
    RuntimeErrorFeedback,
    SuccessFeedback,
)
from arid_badger.gpu_mode_kernel.kernel_pack import KernelPack, TestArgsT
from arid_badger.gpu_mode_kernel.prompts import (
    build_base_prompt,
    extract_last_python_codeblock,
)
from arid_badger.hill_climbing.domain import Evaluation
from arid_badger.landscape_map.v1.domain import SpeedupBin

from ..domain import speedup_band_for_bin


# ---------------------------------------------------------------------------
# Truncation budgets — copied from
# ``arid_badger.gpu_mode_kernel.prompts``. Five-line helpers; not worth a
# library dep for the purpose of one re-export.
# ---------------------------------------------------------------------------

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


def _delta_description(speedup: float, lo: float, hi: float) -> str:
    """Human-readable description of where ``speedup`` sits relative
    to the target band ``[lo, hi)``.
    """
    if speedup < lo:
        return f"below the band by {lo / speedup:.2f}× (your kernel is too slow)"
    if hi != math.inf and speedup >= hi:
        return f"above the band by {speedup / hi:.2f}× (your kernel is too fast)"
    return "inside the target band"


# Two phrase edits applied to the pack's ``kernel_description_body`` before
# it is slotted into the goal-conditioned template. Without these, the body's
# stock "highly optimized" / "optimize runtime for these" framing fights the
# goal-conditioning signal. The substitutions are no-ops on packs whose body
# happens not to contain the source phrases. Validated under e0088 against
# TriMul's body, which contains both.
_BAND_NEUTRAL_BODY_SUBSTITUTIONS: list[tuple[str, str]] = [
    (
        "expert Triton engineer tasked with translating PyTorch code into highly optimized Triton kernel code",
        "Triton engineer translating PyTorch code into Triton kernel code that hits a target performance band",
    ),
    (
        "Test Cases for correctness and runtime (optimize runtime for these):",
        "Test cases (your kernel will be measured on these for both correctness and runtime):",
    ),
]


def _make_body_band_neutral(base_prompt: str) -> str:
    """Strip max-speed framing phrases from the pack's base prompt so it
    composes cleanly with the goal-conditioning preamble."""
    out = base_prompt
    for source, replacement in _BAND_NEUTRAL_BODY_SUBSTITUTIONS:
        out = out.replace(source, replacement)
    return out


def _per_case_lines(
    feedback: SuccessFeedback[CaseSpeedupT],
    target_mid: float,
) -> list[str]:
    """Format per-case breakdown lines, sorted by log-distance from
    the target midpoint."""
    cases_with_distance: list[tuple[float, str]] = []
    log_target = math.log(target_mid) if target_mid > 0 else 0.0
    for case in feedback.per_case_speedups:
        if case.speedup > 0:
            distance = abs(math.log(case.speedup) - log_target)
        else:
            distance = math.inf
        cases_with_distance.append((distance, case.format_for_prompt()))
    cases_with_distance.sort(key=lambda x: x[0])
    return [line for _d, line in cases_with_distance]


# ---------------------------------------------------------------------------
# Render context + render entry points (pure; used directly by tests).
# ---------------------------------------------------------------------------


def build_render_context(
    *,
    pack: KernelPack[TestArgsT, CaseSpeedupT],
    target_bin: SpeedupBin,
    parent_code: str | None,
    evaluation: Evaluation[GpuModeKernelObservation[CaseSpeedupT]] | None,
    gpu_name: str,
    triton_version: str,
) -> dict[str, Any]:
    """Build the dict passed to ``mutation.jinja``.

    ``parent_code`` and ``evaluation`` are both ``None`` for the
    cold-start / root-mutation path. When ``evaluation`` carries an
    ``InfrastructureFailureFeedback`` we render the no-feedback
    ``"none"`` arm — the harness fault carries no signal worth feeding
    to the LLM, matching the upstream provider's behavior.

    The ``base_prompt`` slot is computed once via
    ``build_base_prompt(pack, …)`` so the pack's kernel-specific
    narrative lives in one place (the pack module), not duplicated
    in the template.
    """
    band = speedup_band_for_bin(target_bin)
    base_prompt = _make_body_band_neutral(
        build_base_prompt(pack, gpu_name=gpu_name, triton_version=triton_version)
    )
    ctx: dict[str, Any] = {
        "target_bin_name": target_bin.name,
        "target_lo": band.lo,
        "target_hi": band.hi,
        "target_mid": band.midpoint,
        "target_display": band.display,
        "gpu_name": gpu_name,
        "triton_version": triton_version,
        "base_prompt": base_prompt,
        "parent_code": parent_code,
        "feedback_kind": "none",
    }

    if parent_code is None or evaluation is None:
        return ctx

    feedback = evaluation.observation.feedback
    if isinstance(feedback, InfrastructureFailureFeedback):
        # Treat infra failures as no-feedback: drop the parent code too,
        # since the harness fault tells us nothing about it.
        ctx["parent_code"] = None
        return ctx

    if isinstance(feedback, SuccessFeedback):
        ctx["feedback_kind"] = "success"
        ctx["aggregated_speedup"] = feedback.aggregated_speedup
        ctx["aggregation_method"] = feedback.aggregation_method
        ctx["delta_description"] = _delta_description(
            feedback.aggregated_speedup, band.lo, band.hi
        )
        ctx["per_case_lines"] = _per_case_lines(feedback, band.midpoint)
    elif isinstance(feedback, CompileFailedFeedback):
        ctx["feedback_kind"] = "compile_failed"
        ctx["compilation_error"] = _truncate_head(
            feedback.compilation_error, _MAX_COMPILATION_ERROR_CHARS
        )
    elif isinstance(feedback, RuntimeErrorFeedback):
        ctx["feedback_kind"] = "runtime_error"
        ctx["runtime_error_name"] = feedback.runtime_error_name
        ctx["runtime_error"] = _truncate_head(
            feedback.runtime_error, _MAX_RUNTIME_ERROR_CHARS
        )
        ctx["traceback"] = _truncate_tail(feedback.traceback, _MAX_TRACEBACK_CHARS)
    elif isinstance(feedback, IncorrectFeedback):
        ctx["feedback_kind"] = "incorrect"
        ctx["error_message"] = _truncate_head(
            feedback.error_message, _MAX_INCORRECT_ERROR_CHARS
        )
    return ctx


def _make_jinja_env() -> Environment:
    return Environment(
        loader=PackageLoader(
            "arid_badger.eval_dataset_builder.v1.goal_conditioned_mutation",
            "templates",
        ),
        autoescape=select_autoescape(disabled_extensions=("jinja",), default=False),
        undefined=StrictUndefined,
        keep_trailing_newline=True,
    )


_jinja_env_singleton: Environment | None = None


def _jinja_env() -> Environment:
    global _jinja_env_singleton
    if _jinja_env_singleton is None:
        _jinja_env_singleton = _make_jinja_env()
    return _jinja_env_singleton


def render_prompt(
    *,
    pack: KernelPack[TestArgsT, CaseSpeedupT],
    target_bin: SpeedupBin,
    parent_code: str | None,
    evaluation: Evaluation[GpuModeKernelObservation[CaseSpeedupT]] | None,
    gpu_name: str,
    triton_version: str,
) -> str:
    """Render the goal-conditioned mutation prompt. Pure."""
    template = _jinja_env().get_template("mutation.jinja")
    return template.render(
        build_render_context(
            pack=pack,
            target_bin=target_bin,
            parent_code=parent_code,
            evaluation=evaluation,
            gpu_name=gpu_name,
            triton_version=triton_version,
        )
    )


# ---------------------------------------------------------------------------
# Per-candidate failure signal — copied verbatim from the upstream
# provider so the v2 driver's ``MutationFailed`` path behaves the same
# under either provider.
# ---------------------------------------------------------------------------


class MutationError(RuntimeError):
    """Raised inside the provider's coroutine to signal a per-candidate
    failure. The v2 driver catches this and emits ``MutationFailed``."""


# ---------------------------------------------------------------------------
# Provider — lifecycle + submit + LLM call copied structurally from
# ``GpuModeKernelFeedbackMutationProvider``.
# ---------------------------------------------------------------------------


class GoalConditionedMutationProvider(Generic[TestArgsT, CaseSpeedupT]):
    """Per-candidate async mutation provider whose prompt is goal-conditioned
    on a target ``SpeedupBin``.

    Implements ``AsyncMutationProvider[GpuModeKernelObservation[CaseSpeedupT]]``.
    Lifecycle (asyncio loop on a daemon thread, semaphore-bounded fan-out,
    context-manager pair) and the litellm submit/extract path are
    structurally identical to the upstream
    ``GpuModeKernelFeedbackMutationProvider`` — the only difference is
    that ``_build_prompt`` renders ``mutation.jinja`` instead of
    composing strings via ``arid_badger.gpu_mode_kernel.prompts``.

    ``max_tokens=None`` is the right setting for Gemini 3 Flash (which
    rejects an explicit cap); Together gpt-oss requires an explicit
    32k cap.
    """

    def __init__(
        self,
        *,
        pack: KernelPack[TestArgsT, CaseSpeedupT],
        target_bin: SpeedupBin,
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
        if target_bin is SpeedupBin.FAILURE:
            raise ValueError(
                "target_bin=FAILURE is not a meaningful goal — search "
                "cannot aim at a non-speedup bin."
            )
        self._pack = pack
        self._target_bin = target_bin
        self._model_slug = model_slug
        self._gpu_name = gpu_name
        self._triton_version = triton_version
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
            name=f"{self._pack.name}-goal-cond-{self._target_bin.name}-mutation-loop",
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
        return render_prompt(
            pack=self._pack,
            target_bin=self._target_bin,
            parent_code=parent_code,
            evaluation=evaluation,
            gpu_name=self._gpu_name,
            triton_version=self._triton_version,
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
                    "{name}/{bin} mutation LLM call failed: {exc}\n{tb}",
                    name=self._pack.name,
                    bin=self._target_bin.name,
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
