"""Tests for TriMulFeedbackMutationProvider and format_trimul_feedback_mutation_prompt.

Unit tests (no marks) verify prompt formatting logic only — no LLM, no GPU.
Integration tests (@pytest.mark.integration) make real LLM calls.
E2E tests (@pytest.mark.integration @pytest.mark.modal) make real LLM
calls and then benchmark the resulting kernel on a Modal GPU.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any
from unittest.mock import patch

import pytest
from loguru import logger
from pydantic import BaseModel

from arid_badger.hill_climbing.domain import Evaluation
from arid_badger.hill_climbing.mutation_providers.trimul_feedback_mutation import (
    TriMulFeedbackMutationProvider,
    TriMulFeedbackMutationRecord,
    _build_base_prompt,
    _extract_last_python_codeblock,
    format_trimul_feedback_mutation_prompt,
    _TRIMUL_BASE_PROMPT,
)
from arid_badger.hill_climbing.scoring_providers.trimul import TriMulObservation
from arid_badger.trimul.core import (
    CaseSpeedup,
    CompileFailedFeedback,
    InfrastructureFailureFeedback,
    IncorrectFeedback,
    RuntimeErrorFeedback,
    SuccessFeedback,
)
from arid_badger.typing_utils import is_ok


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# A simple but functionally correct PyTorch implementation of custom_kernel.
# Used as the parent program in integration / e2e tests.
_STARTER_KERNEL = """\
import torch
from torch import einsum
import torch.nn.functional as F


def custom_kernel(data):
    input_tensor, mask, weights, config = data
    x = F.layer_norm(
        input_tensor, [config["dim"]], weights["norm.weight"], weights["norm.bias"]
    )
    mask_ = mask.unsqueeze(-1)
    left = (x @ weights["left_proj.weight"].T) * mask_ * torch.sigmoid(
        x @ weights["left_gate.weight"].T
    )
    right = (x @ weights["right_proj.weight"].T) * mask_ * torch.sigmoid(
        x @ weights["right_gate.weight"].T
    )
    out = einsum("...ikd,...jkd->...ijd", left, right)
    out = F.layer_norm(
        out,
        [config["hidden_dim"]],
        weights["to_out_norm.weight"],
        weights["to_out_norm.bias"],
    )
    out = out * torch.sigmoid(x @ weights["out_gate.weight"].T)
    return out @ weights["to_out.weight"].T
"""

_MODEL_SLUG = "gemini/gemini-2.5-flash"


class _ListSink:
    """In-memory InvocationSink that keeps a list of records."""

    def __init__(self) -> None:
        self.records: list[BaseModel] = []

    def record(self, payload: BaseModel) -> None:
        self.records.append(payload)


def _success_eval(
    speedup: float = 1.0,
    per_case_speedups: list[CaseSpeedup] | None = None,
) -> Evaluation[TriMulObservation]:
    """Build a SuccessFeedback evaluation with a self-consistent single case.

    Holds candidate ``runtime_ns`` constant at 1e6 and varies
    ``ref_runtime_ns = 1e6 * speedup`` so the invariant
    ``speedup == ref_runtime_ns / runtime_ns`` (see ``trimul_modal.py``)
    holds for any ``speedup`` value. The pattern reads as if the dimensions
    are inverted, but the math is correct: at speedup=2.0 the reference
    is 2 ms and the candidate is 1 ms, i.e. the candidate is 2x faster.
    """
    if per_case_speedups is None:
        per_case_speedups = [
            CaseSpeedup(
                seqlen=256,
                bs=2,
                dim=128,
                hiddendim=128,
                nomask=True,
                distribution="normal",
                speedup=speedup,
                runtime_ns=1_000_000.0,
                ref_runtime_ns=1_000_000.0 * speedup,
            )
        ]
    return Evaluation[TriMulObservation](
        observation=TriMulObservation(
            feedback=SuccessFeedback(
                aggregated_speedup=speedup,
                aggregation_method="geomean",
                per_case_speedups=per_case_speedups,
            )
        ),
        reward=speedup,
    )


def _infra_failure_eval() -> Evaluation[TriMulObservation]:
    return Evaluation[TriMulObservation](
        observation=TriMulObservation(
            feedback=InfrastructureFailureFeedback(reason="modal eviction")
        ),
        reward=None,
    )


# ---------------------------------------------------------------------------
# Unit tests — prompt formatting only (no LLM, no GPU)
# ---------------------------------------------------------------------------


def test_format_compile_failed_prompt() -> None:
    feedback = CompileFailedFeedback(compilation_error="SyntaxError: bad indent on line 3")
    prompt = format_trimul_feedback_mutation_prompt(
        base_prompt=_TRIMUL_BASE_PROMPT,
        previous_kernel_code=_STARTER_KERNEL,
        feedback=feedback,
    )

    assert "SyntaxError: bad indent on line 3" in prompt
    assert "failed to compile" in prompt
    assert "custom_kernel" in prompt  # starter kernel is included
    assert _TRIMUL_BASE_PROMPT[:50] in prompt  # base prompt prefix present


def test_format_runtime_error_prompt() -> None:
    feedback = RuntimeErrorFeedback(
        runtime_error_name="RuntimeError",
        runtime_error="CUDA error: an illegal memory access was encountered",
        traceback="Traceback (most recent call last):\n  File ...",
    )
    prompt = format_trimul_feedback_mutation_prompt(
        base_prompt=_TRIMUL_BASE_PROMPT,
        previous_kernel_code=_STARTER_KERNEL,
        feedback=feedback,
    )

    assert "RuntimeError" in prompt
    assert "illegal memory access" in prompt
    assert "Traceback" in prompt
    assert "custom_kernel" in prompt


def test_format_incorrect_prompt() -> None:
    error_msg = "Number of mismatched elements: 5\nERROR at (0, 1, 2, 3): 0.12345 0.23456"
    feedback = IncorrectFeedback(error_message=error_msg)
    prompt = format_trimul_feedback_mutation_prompt(
        base_prompt=_TRIMUL_BASE_PROMPT,
        previous_kernel_code=_STARTER_KERNEL,
        feedback=feedback,
    )

    assert "Number of mismatched elements: 5" in prompt
    assert "ERROR at (0, 1, 2, 3)" in prompt
    assert "incorrect output" in prompt
    assert "custom_kernel" in prompt


def test_format_success_prompt_ordering() -> None:
    # Two cases: fast (2.1x) and slow (0.9x). Slow should appear first.
    per_case = [
        CaseSpeedup(
            seqlen=1024, bs=1, dim=384, hiddendim=128,
            nomask=True, distribution="normal",
            speedup=2.1, runtime_ns=500_000.0, ref_runtime_ns=1_050_000.0,
        ),
        CaseSpeedup(
            seqlen=256, bs=2, dim=128, hiddendim=128,
            nomask=False, distribution="cauchy",
            speedup=0.9, runtime_ns=1_200_000.0, ref_runtime_ns=1_080_000.0,
        ),
    ]
    feedback = SuccessFeedback(
        aggregated_speedup=1.5,
        aggregation_method="geomean",
        per_case_speedups=per_case,
    )
    prompt = format_trimul_feedback_mutation_prompt(
        base_prompt=_TRIMUL_BASE_PROMPT,
        previous_kernel_code=_STARTER_KERNEL,
        feedback=feedback,
    )

    assert "1.500x" in prompt
    assert "geomean" in prompt
    # Slow case (0.9x, seqlen=256) must appear before fast case (2.1x, seqlen=1024)
    idx_slow = prompt.index("0.900x")
    idx_fast = prompt.index("2.100x")
    assert idx_slow < idx_fast, "Slow case should appear before fast case in prompt"
    assert "custom_kernel" in prompt


def test_compile_error_head_truncation() -> None:
    # A unique sentinel inside the early portion of the message confirms
    # head-truncation kept the prefix.
    head_marker = "FIRST_LINE_MARKER"
    tail_marker = "LAST_LINE_MARKER"
    long_error = head_marker + ("X" * 50_000) + tail_marker
    feedback = CompileFailedFeedback(compilation_error=long_error)
    prompt = format_trimul_feedback_mutation_prompt(
        base_prompt=_TRIMUL_BASE_PROMPT,
        previous_kernel_code=_STARTER_KERNEL,
        feedback=feedback,
    )

    assert "[truncated]" in prompt
    assert head_marker in prompt
    assert tail_marker not in prompt
    assert long_error not in prompt


def test_runtime_traceback_tail_truncation() -> None:
    # Tracebacks tail-truncate: the deepest frame is at the bottom.
    head_marker = "OLDEST_FRAME"
    tail_marker = "DEEPEST_FRAME"
    long_traceback = head_marker + ("Y" * 50_000) + tail_marker
    feedback = RuntimeErrorFeedback(
        runtime_error_name="RuntimeError",
        runtime_error="boom",
        traceback=long_traceback,
    )
    prompt = format_trimul_feedback_mutation_prompt(
        base_prompt=_TRIMUL_BASE_PROMPT,
        previous_kernel_code=_STARTER_KERNEL,
        feedback=feedback,
    )

    assert "[truncated]" in prompt
    assert tail_marker in prompt, "Expected deepest frame to be preserved"
    assert head_marker not in prompt
    assert long_traceback not in prompt


def test_rules_block_present_when_provider_builds_prompt() -> None:
    base_prompt = _build_base_prompt(gpu_name="H200", triton_version="3.4.0")
    prompt = format_trimul_feedback_mutation_prompt(
        base_prompt=base_prompt,
        previous_kernel_code=_STARTER_KERNEL,
        feedback=CompileFailedFeedback(compilation_error="x"),
    )

    assert "Nvidia H200" in prompt
    assert "triton 3.4.0" in prompt
    assert "Define all of your code in one final ```python ``` block." in prompt
    assert "final output is in float32" in prompt
    assert "Include a short docstring" in prompt


def test_extract_last_python_codeblock_picks_final() -> None:
    text = (
        "Here is a draft attempt:\n"
        "```python\n"
        "def custom_kernel(data):\n"
        "    return data  # DRAFT\n"
        "```\n"
        "On reflection, the final version is:\n"
        "```python\n"
        "def custom_kernel(data):\n"
        "    return data  # FINAL\n"
        "```\n"
    )

    code = _extract_last_python_codeblock(text)

    assert code is not None
    assert "FINAL" in code
    assert "DRAFT" not in code


def test_extract_last_python_codeblock_returns_none_when_missing() -> None:
    assert _extract_last_python_codeblock("no code here") is None
    assert _extract_last_python_codeblock("```python\n\n```") is None


# ---------------------------------------------------------------------------
# Unit tests — fan-out shape (no real LLM)
# ---------------------------------------------------------------------------


class _FakeUsage:
    def __init__(self, prompt: int, completion: int) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion


class _FakeMessage:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _FakeChoice:
    def __init__(self, content: str | None) -> None:
        self.message = _FakeMessage(content)


class _FakeResponse:
    def __init__(self, content: str | None) -> None:
        self.choices = [_FakeChoice(content)]
        self.usage = _FakeUsage(prompt=10, completion=20)


def test_fanout_issues_one_call_per_mutation_with_n1() -> None:
    """num_mutations independent acompletion calls, each with no ``n=`` param.

    Guards against a regression that would reintroduce ``n=num_mutations``
    on the wire — which Gemini rejects with HTTP 400. The fan-out pattern
    must stay one-call-per-candidate.
    """
    provider = TriMulFeedbackMutationProvider(
        model_slug=_MODEL_SLUG,
        gpu_name="A100-80GB",
        max_llm_concurrency=2,
    )

    fake_code = "```python\ndef custom_kernel(data):\n    return data\n```"
    call_kwargs: list[dict[str, Any]] = []

    async def fake_acompletion(**kwargs: Any) -> _FakeResponse:
        call_kwargs.append(kwargs)
        return _FakeResponse(fake_code)

    with patch(
        "arid_badger.hill_climbing.mutation_providers.trimul_feedback_mutation.litellm.acompletion",
        side_effect=fake_acompletion,
    ):
        codes = provider.generate_mutations(
            program_code=_STARTER_KERNEL,
            num_mutations=3,
            evaluation=_infra_failure_eval(),
        )

    assert len(codes) == 3
    assert len(call_kwargs) == 3
    for kwargs in call_kwargs:
        assert "n" not in kwargs, (
            "acompletion must be called with implicit n=1 — Gemini rejects n>1"
        )
        assert kwargs["model"] == _MODEL_SLUG
        assert kwargs["num_retries"] == 4
        assert kwargs["timeout"] == 300.0


def test_fanout_respects_concurrency_limit() -> None:
    """Semaphore caps in-flight acompletion calls to max_llm_concurrency."""
    provider = TriMulFeedbackMutationProvider(
        model_slug=_MODEL_SLUG,
        gpu_name="A100-80GB",
        max_llm_concurrency=2,
    )

    fake_code = "```python\ndef custom_kernel(data):\n    return data\n```"
    in_flight = 0
    peak_in_flight = 0
    lock = asyncio.Lock()

    async def fake_acompletion(**_kwargs: Any) -> _FakeResponse:
        nonlocal in_flight, peak_in_flight
        async with lock:
            in_flight += 1
            peak_in_flight = max(peak_in_flight, in_flight)
        # Yield so other coroutines can run and hit the semaphore.
        await asyncio.sleep(0.01)
        async with lock:
            in_flight -= 1
        return _FakeResponse(fake_code)

    with patch(
        "arid_badger.hill_climbing.mutation_providers.trimul_feedback_mutation.litellm.acompletion",
        side_effect=fake_acompletion,
    ):
        codes = provider.generate_mutations(
            program_code=_STARTER_KERNEL,
            num_mutations=8,
            evaluation=_infra_failure_eval(),
        )

    assert len(codes) == 8
    assert peak_in_flight <= 2, f"peak in-flight {peak_in_flight} exceeded cap 2"


def test_fanout_tolerates_partial_failures() -> None:
    """Failed calls are logged and swallowed; surviving calls produce codes."""
    provider = TriMulFeedbackMutationProvider(
        model_slug=_MODEL_SLUG,
        gpu_name="A100-80GB",
        max_llm_concurrency=4,
    )

    fake_code = "```python\ndef custom_kernel(data):\n    return data\n```"
    call_index = 0
    index_lock = asyncio.Lock()

    async def fake_acompletion(**_kwargs: Any) -> _FakeResponse:
        nonlocal call_index
        async with index_lock:
            this_index = call_index
            call_index += 1
        if this_index % 2 == 0:
            raise RuntimeError(f"simulated failure on call {this_index}")
        return _FakeResponse(fake_code)

    sink = _ListSink()
    provider_with_sink = TriMulFeedbackMutationProvider(
        model_slug=_MODEL_SLUG,
        gpu_name="A100-80GB",
        max_llm_concurrency=4,
        invocation_sink=sink,
    )

    with patch(
        "arid_badger.hill_climbing.mutation_providers.trimul_feedback_mutation.litellm.acompletion",
        side_effect=fake_acompletion,
    ):
        codes = provider_with_sink.generate_mutations(
            program_code=_STARTER_KERNEL,
            num_mutations=4,
            evaluation=_infra_failure_eval(),
        )

    # 2 of 4 calls failed (indices 0 and 2 under the toy counter). We don't
    # rely on that exact count because asyncio scheduling can interleave,
    # but we do require strictly fewer produced than requested and > 0.
    assert 0 < len(codes) < 4
    assert len(sink.records) == 1
    record = sink.records[0]
    assert isinstance(record, TriMulFeedbackMutationRecord)
    assert record.num_mutations_requested == 4
    assert record.num_mutations_produced == len(codes)
    assert record.num_llm_requests == 4
    # Silence unused-import warning now that we no longer reference `provider`.
    del provider


def test_rejects_concurrency_below_one() -> None:
    with pytest.raises(ValueError, match="max_llm_concurrency"):
        TriMulFeedbackMutationProvider(
            model_slug=_MODEL_SLUG,
            gpu_name="A100-80GB",
            max_llm_concurrency=0,
        )


# ---------------------------------------------------------------------------
# Integration tests — real LLM calls, no GPU
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_generate_mutations_success_feedback() -> None:
    """Real LLM call conditioned on SuccessFeedback.

    Verifies the feedback-prompt branch produces parseable kernels and that
    the invocation record is populated correctly.
    """
    sink = _ListSink()
    provider = TriMulFeedbackMutationProvider(
        model_slug=_MODEL_SLUG,
        gpu_name="L4",
        invocation_sink=sink,
    )
    eval_ = _success_eval(speedup=1.0)

    codes = provider.generate_mutations(
        program_code=_STARTER_KERNEL,
        num_mutations=2,
        evaluation=eval_,
    )

    for i, code in enumerate(codes):
        logger.info("success-feedback mutation {i}/{n}:\n{code}", i=i, n=len(codes), code=code)

    assert len(codes) >= 1, "Expected at least one parseable candidate"
    assert len(codes) <= 2
    for code in codes:
        assert "custom_kernel" in code, f"Expected custom_kernel in:\n{code[:300]}"

    assert len(sink.records) == 1
    record = sink.records[0]
    assert isinstance(record, TriMulFeedbackMutationRecord)
    logger.info("invocation record:\n{}", record.model_dump_json(indent=2))
    assert record.num_mutations_requested == 2
    assert record.num_mutations_produced == len(codes)
    assert record.total_input_tokens > 0
    assert record.total_output_tokens > 0
    assert record.wall_clock_seconds > 0
    # One outbound call per requested candidate (n=1 fan-out).
    assert record.num_llm_requests == 2


@pytest.mark.integration
def test_generate_mutations_infrastructure_failure() -> None:
    """Real LLM call with InfrastructureFailureFeedback (zero-shot branch).

    Verifies that when execution feedback is unavailable the provider
    issues the base prompt and still returns parseable kernels.
    """
    sink = _ListSink()
    provider = TriMulFeedbackMutationProvider(
        model_slug=_MODEL_SLUG,
        gpu_name="L4",
        invocation_sink=sink,
    )

    codes = provider.generate_mutations(
        program_code=_STARTER_KERNEL,
        num_mutations=2,
        evaluation=_infra_failure_eval(),
    )

    for i, code in enumerate(codes):
        logger.info("zero-shot mutation {i}/{n}:\n{code}", i=i, n=len(codes), code=code)

    assert len(codes) >= 1
    assert len(codes) <= 2
    for code in codes:
        assert "custom_kernel" in code

    assert len(sink.records) == 1
    record = sink.records[0]
    assert isinstance(record, TriMulFeedbackMutationRecord)
    logger.info("invocation record:\n{}", record.model_dump_json(indent=2))
    assert record.total_input_tokens > 0


# ---------------------------------------------------------------------------
# E2E test — real LLM call → Modal GPU benchmark
# ---------------------------------------------------------------------------


@pytest.mark.integration
@pytest.mark.modal
def test_e2e_mutation_then_modal_benchmark() -> None:
    """Sample one mutation from the provider, then benchmark it on Modal.

    Verifies the full pipeline: LLM generates a candidate → Modal scores it.
    We don't assert correctness or speedup, just that the scoring pipeline
    completed without an infrastructure failure.
    """
    from arid_badger.trimul.cases import BENCHMARK_CASES
    from arid_badger.trimul.modal_scoring import modal_trimul_scoring_session

    provider = TriMulFeedbackMutationProvider(model_slug=_MODEL_SLUG, gpu_name="L4")

    codes = provider.generate_mutations(
        program_code=_STARTER_KERNEL,
        num_mutations=1,
        evaluation=_infra_failure_eval(),
    )

    if not codes:
        pytest.skip("LLM produced no parseable candidates; skipping Modal benchmark.")

    candidate = codes[0]
    logger.info("benchmarking candidate on Modal:\n{}", candidate)

    # Score against one benchmark case to keep wall-clock short.
    modal_start_s = time.perf_counter()
    with modal_trimul_scoring_session(gpu="L4") as score_fn:
        results = score_fn(candidate, BENCHMARK_CASES[:1])
    modal_elapsed_s = time.perf_counter() - modal_start_s

    assert len(results) == 1
    result = results[0]
    # is_ok() returns True if the scoring pipeline ran (even if the kernel
    # is incorrect). Only infrastructure failures produce Err.
    assert is_ok(result), f"Scoring infrastructure failed: {result}"
    exec_result = result.unwrap()
    logger.info(
        "modal scoring done in {elapsed:.2f}s; result:\n{result}",
        elapsed=modal_elapsed_s,
        result=exec_result.model_dump_json(indent=2),
    )
