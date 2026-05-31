"""Causal conv1d mutation provider conditioned on execution feedback.

Near-duplicate of ``trimul_feedback_mutation``. The plumbing (asyncio
fan-out, code-block extraction, truncation budgets, invocation record,
Provider class lifecycle) is generic and slated for the gh070-A task #3
extraction. The two pieces that are genuinely kernel-specific:
- ``_CAUSAL_CONV1D_BASE_PROMPT``: kernel description, PyTorch reference,
  test-case constraints.
- ``format_causal_conv1d_feedback_mutation_prompt``: the success-path
  per-case formatter (uses ``B, D, S, W`` shape fields rather than
  TriMul's ``seqlen, bs, dim, hiddendim, nomask, dist``).

Everything else is a copy. Drift between the two providers is a code
smell — fix it in #3 by extracting the generic skeleton and parameter-
ising on a kernel pack rather than by hand-syncing.
"""

from __future__ import annotations

import asyncio
import re
import time
import traceback
from datetime import datetime, timezone
from typing import Literal

import litellm
from loguru import logger
from pydantic import BaseModel

from gpu_forecasters.causal_conv1d.cases import (
    BENCHMARK_CASES,
    CORRECTNESS_CASES,
)
from gpu_forecasters.causal_conv1d.core import (
    CausalConv1dKernelExecutionFeedback,
    CompileFailedFeedback,
    InfrastructureFailureFeedback,
    IncorrectFeedback,
    RuntimeErrorFeedback,
)
from gpu_forecasters.hill_climbing.domain import Evaluation, MutationProvider
from gpu_forecasters.hill_climbing.scoring_providers.causal_conv1d import (
    CausalConv1dObservation,
)
from gpu_forecasters.invocation_sink import InvocationSink, code_sha256
from gpu_forecasters.typing_utils import implements


# ---------------------------------------------------------------------------
# Base prompt
# ---------------------------------------------------------------------------
#
# Adapted from the TriMul-style compositional prompt path. We follow the
# same structure (problem framing → constraints → reference PyTorch →
# entrypoint skeleton → triton example → rules) so that the prompt-
# engineering shape matches what we already validated on TriMul.
#
# The kernel is a Mamba-style causal depthwise 1D convolution. Each
# channel is convolved independently with causal (left) zero padding so
# that ``out[t]`` depends only on inputs at positions ``[t-W+1 .. t]``.
# The reference is a thin wrapper around ``torch.nn.functional.conv1d``
# with ``groups=D``.

_CAUSAL_CONV1D_BASE_PROMPT_BODY = r'''You are an expert Triton engineer tasked with translating PyTorch code into highly optimized Triton kernel code.

You will be implementing a causal depthwise 1D convolution kernel — a core component of Mamba and Mamba-2 state-space-model architectures, where it sits in the input pre-processing path before the SSM scan.

The operator processes a 3D tensor of shape [B, D, S] (batch, channels, sequence). Each channel is convolved independently (depthwise, groups=D) with a per-channel filter of width W, plus a per-channel bias. The convolution is **causal**: output at time t depends only on inputs at positions [t-W+1 .. t], with out-of-bounds reads treated as zero. Concretely:

  out[b, d, t] = bias[d] + sum_{k=0..W-1} weight[d, k] * x[b, d, t - W + 1 + k]

Your task:
- Implement the forward pass only. No gradients.
- Your function must be defined as ``custom_kernel`` with the signature below.

Input:
- ``data``: tuple ``(x, weight, bias)`` where
    - ``x``      : ``torch.Tensor`` of shape ``[B, D, S]``, float32, on CUDA
    - ``weight`` : ``torch.Tensor`` of shape ``[D, W]``, float32, on CUDA
    - ``bias``   : ``torch.Tensor`` of shape ``[D]``,    float32, on CUDA

Output:
- ``torch.Tensor`` of shape ``[B, D, S]``, float32

**Problem Constraints:**
- B ∈ {1, 2, 4}
- D ∈ {64, 128, 256, 768, 1536, 2560} (depthwise — independent channels, no cross-channel mixing)
- S ∈ {64, 128, 256, 2048, 4096}
- W ∈ {3, 4, 8} (small — the filter width is tiny relative to D and S)
- Inputs are sampled from a standard Normal distribution.

**Remarks.** This kernel is interesting because it's *very* memory-bound: each output element does only W multiply-adds, but reads W+1 inputs and one weight. The win comes from (1) keeping the small per-channel weight in registers across the time dimension, (2) tiling so that each program loads contiguous time strips of x, and (3) avoiding the explicit zero-pad allocation that the PyTorch reference does. The depthwise structure means there is no reduction across D, so D parallelises trivially.

Here is a PyTorch implementation. You should implement a kernel for the operations in ``ref_kernel``:

```python
import torch
import torch.nn.functional as F


def ref_kernel(data):
    x, weight, bias = data
    _, D, _ = x.shape
    W = weight.shape[1]
    # Causal (left) zero-padding so output[t] depends on input[t-W+1:t+1].
    x_padded = F.pad(x, (W - 1, 0))
    # Depthwise conv1d (groups=D).
    return F.conv1d(
        x_padded,
        weight.unsqueeze(1),  # [D, 1, W]
        bias=bias,
        groups=D,
    )
```

Here is some example skeleton code of the entrypoint function you will create:
```python
def custom_kernel(data):
    x, weight, bias = data
    B, D, S = x.shape
    W = weight.shape[1]

    # ... your kernel here ...

    return out  # shape [B, D, S]
```

To help you understand which Triton version we are using, here is some example Triton code for an unrelated task:
```python
import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    c = acc.to(tl.float16)
    c_ptrs = c_ptr + stride_cm * offs_m[:, None] + stride_cn * offs_n[None, :]
    tl.store(c_ptrs, c)
```

A few general triton tips:
- ``tl.arange`` only takes constexpr arguments (static or ``tl.constexpr``)
- You cannot use ``continue`` in your kernel code
- ``tl.dot`` can only take in two input tensors
- There is no ``tl.mean``

Test cases for correctness (small shapes; produced runtime is not optimized for these):
'''

# Append the test/benchmark case lists to the base prompt at module
# load — keeping them here lets the LLM see exactly which shapes its
# kernel will be evaluated on.
def _format_cases_block() -> str:
    lines: list[str] = []
    for c in CORRECTNESS_CASES:
        lines.append(
            f'  - {{"B": {c["B"]}, "D": {c["D"]}, "S": {c["S"]}, "W": {c["W"]}}}'
        )
    lines.append("")
    lines.append("Benchmark cases for runtime (optimize runtime for these):")
    for c in BENCHMARK_CASES:
        lines.append(
            f'  - {{"B": {c["B"]}, "D": {c["D"]}, "S": {c["S"]}, "W": {c["W"]}}}'
        )
    return "\n".join(lines) + "\n"


_CAUSAL_CONV1D_BASE_PROMPT = (
    _CAUSAL_CONV1D_BASE_PROMPT_BODY.rstrip() + "\n" + _format_cases_block()
)


# ---------------------------------------------------------------------------
# Rules block — same structure as TriMul's, parameterised on GPU and
# Triton version. Will be lifted into the shared module in #3.
# ---------------------------------------------------------------------------

_RULES_TEMPLATE = """\
Rules:
- The tensors arguments passed in will be already on your cuda device.
- Define all of your code in one final ```python ``` block.
- We will test the correctness of your kernel on multiple input shapes, make sure to support different potential test cases.
- You are allowed to use mixed precision computations, but make sure your final output is in float32.
- You must use triton {triton_version} and these kernels will be run on an Nvidia {gpu_name}.
- You do not have to implement everything in triton, you may choose to have some of the operations done in pytorch. However, you must implement at least part of the operations in a kernel.
- Include a short docstring at the top summarizing your algorithm.
"""


def _build_base_prompt(*, gpu_name: str, triton_version: str) -> str:
    rules = _RULES_TEMPLATE.format(gpu_name=gpu_name, triton_version=triton_version)
    return _CAUSAL_CONV1D_BASE_PROMPT.rstrip() + "\n\n" + rules


# ---------------------------------------------------------------------------
# Code extraction (kernel-agnostic)
# ---------------------------------------------------------------------------

# Picks the LAST python block — the rules instruct the model to put its
# final code in one trailing block, but reasoning models often emit
# drafts above that final block.
_PYTHON_CODEBLOCK_RE = re.compile(
    r"```python\n(?!```)(.*?)(?:\n```)?(?=\n```|$)",
    re.DOTALL,
)


def _extract_last_python_codeblock(text: str) -> str | None:
    matches = list(_PYTHON_CODEBLOCK_RE.finditer(text))
    if not matches:
        return None
    code = matches[-1].group(1).rstrip()
    return code or None


# ---------------------------------------------------------------------------
# Prompt formatting
# ---------------------------------------------------------------------------

# Per-feedback-kind truncation budgets. Tracebacks tail-truncate (the
# deepest frame is at the bottom and is usually the actionable signal);
# everything else head-truncates (first error line is usually the
# actionable signal).
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


def format_causal_conv1d_feedback_mutation_prompt(
    *,
    base_prompt: str,
    previous_kernel_code: str,
    feedback: CausalConv1dKernelExecutionFeedback,
) -> str:
    """Build a mutation prompt from a base prompt and prior execution feedback.

    ``feedback`` must be one of the four concrete causal conv1d feedback
    types. Callers should pass the base prompt directly (zero-shot) for
    ``InfrastructureFailureFeedback``.
    """
    if not base_prompt:
        raise ValueError("base_prompt must be non-empty.")
    if not previous_kernel_code:
        raise ValueError("previous_kernel_code must be non-empty.")

    prompt = base_prompt.rstrip() + "\n"
    prompt += "\nHere is your latest implementation:\n"
    prompt += f"```python\n{previous_kernel_code}\n```\n\n"
    prompt += "Your custom_kernel was evaluated on GPU.\n\nHere is the evaluation result:\n"

    if isinstance(feedback, CompileFailedFeedback):
        prompt += "Your kernel failed to compile.\n\n"
        prompt += "Compilation error:\n"
        prompt += f"{_truncate_head(feedback.compilation_error, _MAX_COMPILATION_ERROR_CHARS)}\n\n"
        prompt += "Please fix the errors and try again."
    elif isinstance(feedback, RuntimeErrorFeedback):
        prompt += "Your kernel raised an exception at runtime.\n\n"
        prompt += f"Error type: {feedback.runtime_error_name}\n\n"
        prompt += "Error message:\n"
        prompt += f"{_truncate_head(feedback.runtime_error, _MAX_RUNTIME_ERROR_CHARS)}\n\n"
        prompt += "Traceback:\n"
        prompt += f"{_truncate_tail(feedback.traceback, _MAX_TRACEBACK_CHARS)}\n\n"
        prompt += "Please fix the errors and try again."
    elif isinstance(feedback, IncorrectFeedback):
        prompt += "Your kernel produced incorrect output compared to the reference.\n\n"
        prompt += "Correctness issue:\n"
        prompt += f"{_truncate_head(feedback.error_message, _MAX_INCORRECT_ERROR_CHARS)}\n\n"
        prompt += "Please fix the correctness issues and try again."
    else:
        # SuccessFeedback — the only remaining variant
        prompt += "You are iteratively optimizing runtime (microseconds).\n\n"
        prompt += (
            f"Your kernel is correct. "
            f"Aggregated speedup: {feedback.aggregated_speedup:.3f}x "
            f"(aggregation method: {feedback.aggregation_method}).\n\n"
        )
        sorted_cases = sorted(feedback.per_case_speedups, key=lambda c: c.speedup)
        prompt += "Per-case breakdown (slowest first):\n"
        for case in sorted_cases:
            ref_us = case.ref_runtime_ns / 1_000.0
            candidate_us = case.runtime_ns / 1_000.0
            prompt += (
                f"  B={case.B}, D={case.D}, S={case.S}, W={case.W}: "
                f"{case.speedup:.3f}x "
                f"(ref: {ref_us:.1f}μs, candidate: {candidate_us:.1f}μs)\n"
            )
        prompt += (
            "\nPlease rewrite the entire kernel to be as fast as possible. "
            "Focus on the slowest configurations listed above."
        )

    return prompt


# ---------------------------------------------------------------------------
# Invocation record
# ---------------------------------------------------------------------------


class CausalConv1dFeedbackMutationRecord(BaseModel, frozen=True):
    """Invocation record for one ``generate_mutations`` call."""

    kind: Literal[
        "causal_conv1d_feedback_mutation"
    ] = "causal_conv1d_feedback_mutation"
    parent_code_sha256: str
    child_code_sha256s: list[str]
    model_slug: str
    total_input_tokens: int
    total_output_tokens: int
    num_mutations_requested: int
    num_mutations_produced: int
    num_llm_requests: int
    wall_clock_seconds: float
    timestamp_utc: str


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class CausalConv1dFeedbackMutationProvider:
    """Generates causal conv1d kernel mutations conditioned on execution
    feedback.

    Implements ``MutationProvider[CausalConv1dObservation]``. Mirrors
    ``TriMulFeedbackMutationProvider`` 1:1; see that provider's
    docstring for the rationale behind ``n=1`` fan-out and the
    ``max_tokens`` knob.
    """

    _model_slug: str
    _base_prompt: str
    _max_llm_concurrency: int
    _num_retries: int
    _request_timeout_s: float
    _max_tokens: int | None
    _invocation_sink: InvocationSink | None

    def __init__(
        self,
        *,
        model_slug: str,
        gpu_name: str,
        triton_version: str = "3.3.1",
        max_llm_concurrency: int = 8,
        num_retries: int = 4,
        request_timeout_s: float = 300.0,
        max_tokens: int | None = None,
        invocation_sink: InvocationSink | None = None,
    ) -> None:
        if max_llm_concurrency < 1:
            raise ValueError("max_llm_concurrency must be >= 1")
        if max_tokens is not None and max_tokens < 1:
            raise ValueError("max_tokens must be >= 1 when set")
        self._model_slug = model_slug
        self._base_prompt = _build_base_prompt(
            gpu_name=gpu_name, triton_version=triton_version
        )
        self._max_llm_concurrency = max_llm_concurrency
        self._num_retries = num_retries
        self._request_timeout_s = request_timeout_s
        self._max_tokens = max_tokens
        self._invocation_sink = invocation_sink

    def generate_mutations(
        self,
        program_code: str,
        num_mutations: int,
        evaluation: Evaluation[CausalConv1dObservation],
    ) -> list[str]:
        feedback = evaluation.observation.feedback
        if isinstance(feedback, InfrastructureFailureFeedback):
            prompt = self._base_prompt
        else:
            prompt = format_causal_conv1d_feedback_mutation_prompt(
                base_prompt=self._base_prompt,
                previous_kernel_code=program_code,
                feedback=feedback,
            )

        parent_sha = code_sha256(program_code)
        logger.info(
            "Causal conv1d mutation phase starting: parent_sha={parent}, "
            "n={n}, concurrency={c}",
            parent=parent_sha[:8],
            n=num_mutations,
            c=self._max_llm_concurrency,
        )
        start_time_s = time.perf_counter()
        codes, input_tokens, output_tokens = asyncio.run(
            self._generate_async(prompt=prompt, n=num_mutations)
        )
        wall_clock_seconds = time.perf_counter() - start_time_s
        logger.info(
            "Causal conv1d mutation phase done: parent_sha={parent}, "
            "produced={produced}/{n}, elapsed={elapsed:.1f}s, in_tok={inp}, "
            "out_tok={out}",
            parent=parent_sha[:8],
            produced=len(codes),
            n=num_mutations,
            elapsed=wall_clock_seconds,
            inp=input_tokens,
            out=output_tokens,
        )

        if self._invocation_sink is not None:
            self._invocation_sink.record(
                CausalConv1dFeedbackMutationRecord(
                    parent_code_sha256=parent_sha,
                    child_code_sha256s=[code_sha256(c) for c in codes],
                    model_slug=self._model_slug,
                    total_input_tokens=input_tokens,
                    total_output_tokens=output_tokens,
                    num_mutations_requested=num_mutations,
                    num_mutations_produced=len(codes),
                    num_llm_requests=num_mutations,
                    wall_clock_seconds=wall_clock_seconds,
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                )
            )

        return codes

    async def _generate_async(
        self, *, prompt: str, n: int
    ) -> tuple[list[str], int, int]:
        semaphore = asyncio.Semaphore(self._max_llm_concurrency)

        async def _single_call(index: int) -> tuple[str | None, int, int]:
            async with semaphore:
                logger.info("LLM call {index}/{n}: starting", index=index, n=n)
                call_start_s = time.perf_counter()
                try:
                    if self._max_tokens is None:
                        response = await litellm.acompletion(
                            model=self._model_slug,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=1.0,
                            num_retries=self._num_retries,
                            timeout=self._request_timeout_s,
                        )
                    else:
                        response = await litellm.acompletion(
                            model=self._model_slug,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=1.0,
                            num_retries=self._num_retries,
                            timeout=self._request_timeout_s,
                            max_tokens=self._max_tokens,
                        )
                except Exception:
                    elapsed = time.perf_counter() - call_start_s
                    logger.warning(
                        "LLM call {index}/{n} failed after {elapsed:.1f}s:\n{tb}",
                        index=index,
                        n=n,
                        elapsed=elapsed,
                        tb=traceback.format_exc(),
                    )
                    return None, 0, 0

                elapsed = time.perf_counter() - call_start_s
                raw_usage = response.usage  # pyright: ignore[reportAttributeAccessIssue]
                input_tokens = (
                    raw_usage.prompt_tokens if raw_usage is not None else 0
                )
                output_tokens = (
                    raw_usage.completion_tokens if raw_usage is not None else 0
                )
                content = response.choices[0].message.content  # pyright: ignore[reportAttributeAccessIssue]
                code = _extract_last_python_codeblock(content) if content else None
                if not code:
                    logger.warning(
                        "LLM call {index}/{n}: no code block extracted "
                        "(elapsed={elapsed:.1f}s, in_tok={inp}, out_tok={out}).",
                        index=index,
                        n=n,
                        elapsed=elapsed,
                        inp=input_tokens,
                        out=output_tokens,
                    )
                    return None, input_tokens, output_tokens
                logger.info(
                    "LLM call {index}/{n}: done (elapsed={elapsed:.1f}s, "
                    "in_tok={inp}, out_tok={out}, code_chars={chars})",
                    index=index,
                    n=n,
                    elapsed=elapsed,
                    inp=input_tokens,
                    out=output_tokens,
                    chars=len(code),
                )
                return code, input_tokens, output_tokens

        results = await asyncio.gather(*[_single_call(i) for i in range(n)])
        codes = [code for code, _, _ in results if code is not None]
        total_input_tokens = sum(inp for _, inp, _ in results)
        total_output_tokens = sum(out for _, _, out in results)
        return codes, total_input_tokens, total_output_tokens


implements(MutationProvider[CausalConv1dObservation])(
    CausalConv1dFeedbackMutationProvider
)
