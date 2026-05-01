"""Unit tests for the v2 KernelBench mutation provider.

Three concerns under test:

1. **Prompt rendering** — for each ``KernelExecutionFeedback`` arm and for
   the no-feedback (infrastructure-failure) case, assert that the rendered
   prompt has the structural elements the eval pipeline depends on (the
   ``ModelNew`` skeleton, the reference module, an output-format directive)
   and the right per-arm body. These are coverage assertions, not byte-level
   snapshots — the prose can drift without breaking the contract.

2. **Code extraction** — ``extract_last_python_codeblock`` picks the *last*
   python code block when reasoning models emit multiple blocks; returns
   ``None`` when no block is present.

3. **Asyncio plumbing** — semaphore caps in-flight LLM calls; concurrent
   submits multiplex on one loop thread; failures inside the coroutine
   surface as ``MutationError``.
"""

from __future__ import annotations

import asyncio
import threading
import time
from typing import Any, Callable
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import BaseModel

from arid_badger.hill_climbing.domain import Evaluation
from arid_badger.hill_climbing.scoring_providers.kernelbench import (
    KernelBenchObservation,
)
from arid_badger.invocation_sink import code_sha256
from arid_badger.kernelbench.core import (
    CompileFailedFeedback,
    IncorrectFeedback,
    InfrastructureFailureFeedback,
    RuntimeErrorFeedback,
    SuccessFeedback,
)
from arid_badger.kernelbench.v2.providers.kernel_execution_feedback import (
    KernelBenchFeedbackMutationProvider,
    KernelBenchMutationRecord,
    MutationError,
    extract_last_python_codeblock,
    render_mutation_prompt,
)

PROVIDER_MODULE = (
    "arid_badger.kernelbench.v2.providers.kernel_execution_feedback"
)


_REFERENCE_KERNEL = '''\
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x * self.scale)

scale = 1.5

def get_inputs():
    return [torch.rand(128, 256)]

def get_init_inputs():
    return [scale]
'''

_PARENT_KERNEL = '''\
import torch
import torch.nn as nn

class ModelNew(nn.Module):
    def __init__(self, scale: float) -> None:
        super().__init__()
        self.scale = float(scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(x * self.scale)
'''


# ---------------------------------------------------------------------------
# Prompt rendering — structural coverage per feedback arm
# ---------------------------------------------------------------------------


def _common_assertions(prompt: str) -> None:
    """Every rendered prompt — regardless of feedback arm — must include
    the structural elements the eval pipeline depends on."""
    assert "ModelNew" in prompt
    assert "load_inline" in prompt
    assert "get_inputs()" in prompt
    assert "get_init_inputs()" in prompt
    # The reference module must be inlined verbatim.
    assert "class Model(nn.Module)" in prompt
    # Output-format directive.
    assert "last python code block" in prompt
    # Skeleton example must be visible.
    assert "_ext = load_inline(" in prompt or "load_inline(" in prompt


def test_render_no_feedback_omits_previous_attempt_section() -> None:
    """Infrastructure-failure path: render with feedback=None — the
    'Previous attempt' header must not appear, since there is no prior
    signal worth conditioning on."""
    prompt = render_mutation_prompt(
        reference_kernel_code=_REFERENCE_KERNEL,
        previous_kernel_code=None,
        feedback=None,
    )
    _common_assertions(prompt)
    assert "# Previous attempt" not in prompt
    # No feedback-arm-specific phrases either.
    assert "Compilation failed" not in prompt
    assert "Runtime error" not in prompt
    assert "Output does not match" not in prompt
    assert "Speedup:" not in prompt


def test_render_compile_failed_includes_compiler_output_and_fix_directive() -> None:
    feedback = CompileFailedFeedback(
        compilation_error_name="nvcc_error",
        compilation_error="error: identifier 'foo' is undefined\nat line 42",
    )
    prompt = render_mutation_prompt(
        reference_kernel_code=_REFERENCE_KERNEL,
        previous_kernel_code=_PARENT_KERNEL,
        feedback=feedback,
    )
    _common_assertions(prompt)
    assert "# Previous attempt" in prompt
    assert "Compilation failed" in prompt
    assert "nvcc_error" in prompt
    assert "identifier 'foo' is undefined" in prompt
    assert "Fix the compilation error" in prompt
    # The parent kernel must be shown so the LLM can edit it.
    assert _PARENT_KERNEL.rstrip() in prompt


def test_render_runtime_error_includes_message_traceback_and_fix_directive() -> None:
    feedback = RuntimeErrorFeedback(
        runtime_error_name="CUDA_ERROR_ILLEGAL_ADDRESS",
        runtime_error="an illegal memory access was encountered",
        runtime_error_traceback="Traceback (most recent call last):\n  ... long ...\n  RuntimeError",
    )
    prompt = render_mutation_prompt(
        reference_kernel_code=_REFERENCE_KERNEL,
        previous_kernel_code=_PARENT_KERNEL,
        feedback=feedback,
    )
    _common_assertions(prompt)
    assert "Runtime error" in prompt
    assert "CUDA_ERROR_ILLEGAL_ADDRESS" in prompt
    assert "an illegal memory access was encountered" in prompt
    assert "Traceback (most recent call last)" in prompt
    assert "Fix the runtime error" in prompt


def test_render_incorrect_includes_diff_stats_and_correctness_directive() -> None:
    feedback = IncorrectFeedback(
        correctness_issue="output tensor max diff exceeds atol",
        max_difference=["0.5", "0.7"],
        avg_difference=["0.1", "0.2"],
    )
    prompt = render_mutation_prompt(
        reference_kernel_code=_REFERENCE_KERNEL,
        previous_kernel_code=_PARENT_KERNEL,
        feedback=feedback,
    )
    _common_assertions(prompt)
    assert "Output does not match" in prompt
    assert "output tensor max diff exceeds atol" in prompt
    assert "0.5" in prompt and "0.7" in prompt
    assert "0.1" in prompt and "0.2" in prompt
    assert "match the reference within standard tolerance" in prompt


def test_render_success_includes_speedup_and_rewrite_directive() -> None:
    feedback = SuccessFeedback(
        runtime_us=42.5,
        ref_runtime_us=170.0,
        speedup=4.0,
    )
    prompt = render_mutation_prompt(
        reference_kernel_code=_REFERENCE_KERNEL,
        previous_kernel_code=_PARENT_KERNEL,
        feedback=feedback,
    )
    _common_assertions(prompt)
    assert "Speedup: 4.000x" in prompt
    assert "42.5" in prompt and "170.0" in prompt
    # The success arm should explicitly invite a more aggressive rewrite,
    # not a tweak of the existing kernel.
    assert "Rewrite" in prompt or "rewrite" in prompt
    assert "more aggressive" in prompt


def test_render_truncates_long_compile_error() -> None:
    """A pathologically long compiler dump must be truncated. The
    truncation marker is part of the contract — readers (and the LLM)
    need to know they are seeing a head, not the whole thing."""
    long_error = "X" * 5000
    feedback = CompileFailedFeedback(
        compilation_error_name="nvcc_error",
        compilation_error=long_error,
    )
    prompt = render_mutation_prompt(
        reference_kernel_code=_REFERENCE_KERNEL,
        previous_kernel_code=_PARENT_KERNEL,
        feedback=feedback,
    )
    assert "[truncated]" in prompt
    # The full error string must not be embedded.
    assert long_error not in prompt


def test_render_requires_previous_code_when_feedback_is_provided() -> None:
    """A feedback section without the kernel it refers to is incoherent —
    the renderer must reject it loudly rather than emit a half-formed
    prompt."""
    feedback = SuccessFeedback(runtime_us=1.0, ref_runtime_us=2.0, speedup=2.0)
    with pytest.raises(ValueError, match="previous_kernel_code is required"):
        render_mutation_prompt(
            reference_kernel_code=_REFERENCE_KERNEL,
            previous_kernel_code=None,
            feedback=feedback,
        )


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------


def test_extract_last_block_picks_trailing_block_after_drafts() -> None:
    """Reasoning models often emit drafts above the final block. The
    extractor must pick the *last* block, not the first."""
    response = (
        "Let me think about this.\n\n"
        "Here is a first draft:\n\n"
        "```python\n"
        "# draft 1\n"
        "class ModelNew: pass\n"
        "```\n\n"
        "Actually, this is better:\n\n"
        "```python\n"
        "# draft 2\n"
        "class ModelNew:\n"
        "    pass\n"
        "```\n"
    )
    code = extract_last_python_codeblock(response)
    assert code is not None
    assert "draft 2" in code
    assert "draft 1" not in code


def test_extract_returns_none_when_no_python_block() -> None:
    """A bare prose response (or a response with only generic ``` blocks
    that aren't python-tagged) yields no extractable code."""
    assert extract_last_python_codeblock("just some words") is None
    assert (
        extract_last_python_codeblock("```\nplain block, no python tag\n```")
        is None
    )


def test_extract_handles_unterminated_trailing_block() -> None:
    """Some providers truncate mid-block. The regex tolerates a missing
    closing fence by falling through to end-of-string."""
    response = "```python\nclass ModelNew: pass\n# no closing fence"
    code = extract_last_python_codeblock(response)
    assert code is not None
    assert "ModelNew" in code


# ---------------------------------------------------------------------------
# Asyncio plumbing — patch litellm.acompletion
# ---------------------------------------------------------------------------


def _success_feedback_evaluation() -> Evaluation[KernelBenchObservation]:
    return Evaluation[KernelBenchObservation](
        observation=KernelBenchObservation(
            feedback=SuccessFeedback(runtime_us=1.0, ref_runtime_us=2.0, speedup=2.0),
        ),
        reward=2.0,
    )


def _infra_failure_evaluation() -> Evaluation[KernelBenchObservation]:
    return Evaluation[KernelBenchObservation](
        observation=KernelBenchObservation(
            feedback=InfrastructureFailureFeedback(reason="modal timeout"),
        ),
        reward=None,
    )


def _make_response(content: str) -> MagicMock:
    """Stand-in for the litellm.ModelResponse shape we read from."""
    choice = MagicMock()
    choice.message.content = content
    response = MagicMock()
    response.choices = [choice]
    return response


def test_lifecycle_starts_and_stops_loop_thread() -> None:
    provider = KernelBenchFeedbackMutationProvider(
        reference_kernel_code=_REFERENCE_KERNEL,
        model_slug="fake/model",
        max_llm_concurrency=2,
    )
    with provider:
        assert provider._loop_thread is not None
        assert provider._loop_thread.is_alive()
        assert provider._loop is not None

    assert provider._loop is None
    assert provider._loop_thread is None
    assert provider._semaphore is None


def test_submit_before_enter_raises() -> None:
    provider = KernelBenchFeedbackMutationProvider(
        reference_kernel_code=_REFERENCE_KERNEL,
        model_slug="fake/model",
    )
    with pytest.raises(RuntimeError, match="must be entered as a context manager"):
        provider.submit(_PARENT_KERNEL, _success_feedback_evaluation())


def test_invalid_max_llm_concurrency_raises() -> None:
    with pytest.raises(ValueError, match="max_llm_concurrency must be >= 1"):
        KernelBenchFeedbackMutationProvider(
            reference_kernel_code=_REFERENCE_KERNEL,
            model_slug="fake/model",
            max_llm_concurrency=0,
        )


def test_invalid_max_tokens_raises() -> None:
    with pytest.raises(ValueError, match="max_tokens must be >= 1 when set"):
        KernelBenchFeedbackMutationProvider(
            reference_kernel_code=_REFERENCE_KERNEL,
            model_slug="fake/model",
            max_tokens=0,
        )


def test_empty_reference_kernel_raises() -> None:
    with pytest.raises(ValueError, match="reference_kernel_code must be non-empty"):
        KernelBenchFeedbackMutationProvider(
            reference_kernel_code="",
            model_slug="fake/model",
        )


def test_submit_success_returns_extracted_code() -> None:
    """Happy path: acompletion returns content with one trailing python
    block; the future resolves to the extracted code."""
    response = _make_response(
        "Reasoning blah.\n\n```python\nclass ModelNew:\n    pass\n```"
    )
    with patch(
        f"{PROVIDER_MODULE}.litellm.acompletion",
        new=AsyncMock(return_value=response),
    ):
        with KernelBenchFeedbackMutationProvider(
            reference_kernel_code=_REFERENCE_KERNEL,
            model_slug="fake/model",
            max_llm_concurrency=2,
        ) as provider:
            future = provider.submit(_PARENT_KERNEL, _success_feedback_evaluation())
            code = future.result(timeout=5.0)

    assert "class ModelNew:" in code


def test_submit_with_infrastructure_failure_renders_no_feedback_prompt() -> None:
    """When the parent's evaluation was an infrastructure failure, the
    prompt must be rendered without the 'Previous attempt' section. We
    verify by capturing the prompt actually sent to acompletion."""
    captured: dict[str, Any] = {}

    async def capture(*args: Any, **kwargs: Any) -> Any:
        captured.update(kwargs)
        return _make_response("```python\nclass ModelNew: pass\n```")

    with patch(
        f"{PROVIDER_MODULE}.litellm.acompletion",
        new=AsyncMock(side_effect=capture),
    ):
        with KernelBenchFeedbackMutationProvider(
            reference_kernel_code=_REFERENCE_KERNEL,
            model_slug="fake/model",
        ) as provider:
            provider.submit(_PARENT_KERNEL, _infra_failure_evaluation()).result(
                timeout=5.0
            )

    prompt = captured["messages"][0]["content"]
    assert "# Previous attempt" not in prompt


def test_submit_empty_content_raises_mutation_error() -> None:
    with patch(
        f"{PROVIDER_MODULE}.litellm.acompletion",
        new=AsyncMock(return_value=_make_response("")),
    ):
        with KernelBenchFeedbackMutationProvider(
            reference_kernel_code=_REFERENCE_KERNEL,
            model_slug="fake/model",
        ) as provider:
            future = provider.submit(_PARENT_KERNEL, _success_feedback_evaluation())
            with pytest.raises(MutationError, match="empty content"):
                future.result(timeout=5.0)


def test_submit_no_code_block_raises_mutation_error() -> None:
    with patch(
        f"{PROVIDER_MODULE}.litellm.acompletion",
        new=AsyncMock(return_value=_make_response("just prose, no code")),
    ):
        with KernelBenchFeedbackMutationProvider(
            reference_kernel_code=_REFERENCE_KERNEL,
            model_slug="fake/model",
        ) as provider:
            future = provider.submit(_PARENT_KERNEL, _success_feedback_evaluation())
            with pytest.raises(MutationError, match="no python code block"):
                future.result(timeout=5.0)


def test_submit_acompletion_raises_becomes_mutation_error() -> None:
    """Network drops, rate-limit explosions, etc. inside acompletion must
    surface as MutationError so the v2 driver logs MutationFailed."""

    async def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("provider 503")

    with patch(
        f"{PROVIDER_MODULE}.litellm.acompletion",
        new=AsyncMock(side_effect=boom),
    ):
        with KernelBenchFeedbackMutationProvider(
            reference_kernel_code=_REFERENCE_KERNEL,
            model_slug="fake/model",
        ) as provider:
            future = provider.submit(_PARENT_KERNEL, _success_feedback_evaluation())
            with pytest.raises(MutationError, match="litellm.acompletion failed"):
                future.result(timeout=5.0)


# ---------------------------------------------------------------------------
# Concurrency: semaphore caps in-flight LLM calls
# ---------------------------------------------------------------------------


def _wait_for(predicate: Callable[[], bool], *, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.01)
    raise AssertionError(f"predicate never became True within {timeout}s")


def test_max_llm_concurrency_caps_concurrent_acompletion_calls() -> None:
    """``max_llm_concurrency=K`` with N>K submits must show peak in-flight
    equal to K — proving the asyncio.Semaphore actually bounds outbound
    calls rather than letting all coroutines run together."""
    max_llm_concurrency = 2
    n_submits = 5

    in_flight = 0
    peak_in_flight = 0
    counter_lock = threading.Lock()

    async def slow_acompletion(*args: Any, **kwargs: Any) -> Any:
        nonlocal in_flight, peak_in_flight
        with counter_lock:
            in_flight += 1
            peak_in_flight = max(peak_in_flight, in_flight)
        try:
            await release_event.wait()
        finally:
            with counter_lock:
                in_flight -= 1
        return _make_response("```python\nclass ModelNew: pass\n```")

    with patch(
        f"{PROVIDER_MODULE}.litellm.acompletion",
        new=AsyncMock(side_effect=slow_acompletion),
    ):
        with KernelBenchFeedbackMutationProvider(
            reference_kernel_code=_REFERENCE_KERNEL,
            model_slug="fake/model",
            max_llm_concurrency=max_llm_concurrency,
        ) as provider:
            assert provider._loop is not None

            async def _make_event() -> asyncio.Event:
                return asyncio.Event()

            release_event = asyncio.run_coroutine_threadsafe(
                _make_event(), provider._loop
            ).result(timeout=5.0)

            futures = [
                provider.submit(_PARENT_KERNEL, _success_feedback_evaluation())
                for _ in range(n_submits)
            ]

            _wait_for(
                lambda: in_flight >= max_llm_concurrency,
                timeout=5.0,
            )
            # Hold long enough to confirm no further coroutines slip past
            # the semaphore.
            time.sleep(0.2)

            with counter_lock:
                observed_peak = peak_in_flight

            provider._loop.call_soon_threadsafe(release_event.set)
            results = [f.result(timeout=10.0) for f in futures]

    assert observed_peak == max_llm_concurrency, (
        f"Expected peak concurrency {max_llm_concurrency}, observed {observed_peak}. "
        "Semaphore is not bounding in-flight LLM calls."
    )
    assert len(results) == n_submits
    for code in results:
        assert "ModelNew" in code


# ---------------------------------------------------------------------------
# Invocation sink — one record per submit, success and failure paths
# ---------------------------------------------------------------------------


class _ListSink:
    """List-backed ``InvocationSink`` for tests — captures records in
    insertion order so a test can assert on counts and field values."""

    def __init__(self) -> None:
        self.records: list[BaseModel] = []

    def record(self, payload: BaseModel) -> None:
        self.records.append(payload)


def _make_response_with_usage(
    content: str, *, prompt_tokens: int, completion_tokens: int
) -> MagicMock:
    """``litellm.ModelResponse`` stand-in with usage populated."""
    choice = MagicMock()
    choice.message.content = content
    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    response = MagicMock()
    response.choices = [choice]
    response.usage = usage
    return response


def test_sink_records_success_with_token_counts_and_shas() -> None:
    """Successful mutation: one record carrying parent+child shas, model,
    tokens, and ``failure_reason=None``."""
    sink = _ListSink()
    response = _make_response_with_usage(
        "Reasoning blah.\n\n```python\nclass ModelNew:\n    pass\n```",
        prompt_tokens=1234,
        completion_tokens=567,
    )
    expected_child_code = "class ModelNew:\n    pass"

    with patch(
        f"{PROVIDER_MODULE}.litellm.acompletion",
        new=AsyncMock(return_value=response),
    ):
        with KernelBenchFeedbackMutationProvider(
            reference_kernel_code=_REFERENCE_KERNEL,
            model_slug="fake/model",
            max_llm_concurrency=2,
            invocation_sink=sink,
        ) as provider:
            provider.submit(
                _PARENT_KERNEL, _success_feedback_evaluation()
            ).result(timeout=5.0)

    assert len(sink.records) == 1
    record = sink.records[0]
    assert isinstance(record, KernelBenchMutationRecord)
    assert record.parent_code_sha256 == code_sha256(_PARENT_KERNEL)
    assert record.child_code_sha256 == code_sha256(expected_child_code)
    assert record.model_slug == "fake/model"
    assert record.input_tokens == 1234
    assert record.output_tokens == 567
    assert record.failure_reason is None
    assert record.wall_clock_seconds >= 0.0


def test_sink_records_failure_when_acompletion_raises() -> None:
    """Network failure inside ``acompletion``: record carries
    ``child_code_sha256=None``, populated ``failure_reason``, and
    ``input_tokens=output_tokens=None`` (no usage info to read)."""
    sink = _ListSink()

    async def boom(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("provider 503")

    with patch(
        f"{PROVIDER_MODULE}.litellm.acompletion",
        new=AsyncMock(side_effect=boom),
    ):
        with KernelBenchFeedbackMutationProvider(
            reference_kernel_code=_REFERENCE_KERNEL,
            model_slug="fake/model",
            invocation_sink=sink,
        ) as provider:
            future = provider.submit(
                _PARENT_KERNEL, _success_feedback_evaluation()
            )
            with pytest.raises(MutationError):
                future.result(timeout=5.0)

    assert len(sink.records) == 1
    record = sink.records[0]
    assert isinstance(record, KernelBenchMutationRecord)
    assert record.child_code_sha256 is None
    assert record.input_tokens is None
    assert record.output_tokens is None
    assert record.failure_reason is not None
    assert "litellm.acompletion failed" in record.failure_reason


def test_sink_records_failure_when_no_code_block() -> None:
    """LLM returned a response with usage info but no python block:
    record has ``child_code_sha256=None``, populated ``failure_reason``,
    *and* preserves token counts (the call was billed)."""
    sink = _ListSink()
    response = _make_response_with_usage(
        "just prose, no code", prompt_tokens=10, completion_tokens=5
    )

    with patch(
        f"{PROVIDER_MODULE}.litellm.acompletion",
        new=AsyncMock(return_value=response),
    ):
        with KernelBenchFeedbackMutationProvider(
            reference_kernel_code=_REFERENCE_KERNEL,
            model_slug="fake/model",
            invocation_sink=sink,
        ) as provider:
            future = provider.submit(
                _PARENT_KERNEL, _success_feedback_evaluation()
            )
            with pytest.raises(MutationError):
                future.result(timeout=5.0)

    assert len(sink.records) == 1
    record = sink.records[0]
    assert isinstance(record, KernelBenchMutationRecord)
    assert record.child_code_sha256 is None
    assert record.input_tokens == 10
    assert record.output_tokens == 5
    assert record.failure_reason is not None
    assert "no python code block" in record.failure_reason


def test_sink_one_record_per_submit_under_concurrency() -> None:
    """N concurrent submits → exactly N records; no drops under the
    semaphore-bounded loop multiplexing."""
    sink = _ListSink()
    n_submits = 6
    response = _make_response_with_usage(
        "```python\nclass ModelNew: pass\n```",
        prompt_tokens=1,
        completion_tokens=1,
    )
    with patch(
        f"{PROVIDER_MODULE}.litellm.acompletion",
        new=AsyncMock(return_value=response),
    ):
        with KernelBenchFeedbackMutationProvider(
            reference_kernel_code=_REFERENCE_KERNEL,
            model_slug="fake/model",
            max_llm_concurrency=3,
            invocation_sink=sink,
        ) as provider:
            futures = [
                provider.submit(_PARENT_KERNEL, _success_feedback_evaluation())
                for _ in range(n_submits)
            ]
            for f in futures:
                f.result(timeout=10.0)

    assert len(sink.records) == n_submits


def test_no_sink_no_error_on_success_or_failure() -> None:
    """Without a sink, both success and failure paths must complete
    without raising (the sink is purely optional)."""
    response = _make_response_with_usage(
        "```python\nclass ModelNew: pass\n```",
        prompt_tokens=1,
        completion_tokens=1,
    )
    with patch(
        f"{PROVIDER_MODULE}.litellm.acompletion",
        new=AsyncMock(return_value=response),
    ):
        with KernelBenchFeedbackMutationProvider(
            reference_kernel_code=_REFERENCE_KERNEL,
            model_slug="fake/model",
            invocation_sink=None,
        ) as provider:
            code = provider.submit(
                _PARENT_KERNEL, _success_feedback_evaluation()
            ).result(timeout=5.0)
            assert "ModelNew" in code
