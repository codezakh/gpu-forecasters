"""TriMul mutation provider conditioned on execution feedback.

Uses the fixed ``_TRIMUL_BASE_PROMPT`` (vendored from ttt-discover rather
than imported) instead of a per-problem parameterised prompt, because the
TriMul task is fixed across all search runs.

Fans out ``num_mutations`` independent ``litellm.acompletion(..., n=1)``
calls concurrently, gated by an ``asyncio.Semaphore``. The shape mirrors
the canonical approach adopted by ``WakePhaseMutationProvider`` (used in
e0019/e0020) and the standalone candidate pool in e0021 — both arrived
at ``n=1`` fan-out independently because Gemini (our default model)
rejects ``n>1`` via litellm with a 400 ``"Multiple candidates is not
enabled for this model"`` error.

Robustness knobs (``num_retries``, ``request_timeout_s``) are litellm-
native and absorb transient rate-limit / timeout failures before a call
surfaces as a failure to the provider. The library ``kernel_execution_
feedback.py`` still exists with its original ``n=num_mutations`` design
and should be considered unused against Gemini until rewritten.
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

from gpu_forecasters.hill_climbing.domain import Evaluation, MutationProvider
from gpu_forecasters.hill_climbing.scoring_providers.trimul import TriMulObservation
from gpu_forecasters.invocation_sink import InvocationSink, code_sha256
from gpu_forecasters.trimul.core import (
    CompileFailedFeedback,
    InfrastructureFailureFeedback,
    IncorrectFeedback,
    RuntimeErrorFeedback,
    TriMulKernelExecutionFeedback,
)
from gpu_forecasters.typing_utils import implements


# ---------------------------------------------------------------------------
# Base prompt — vendored verbatim from
# ttt-discover/examples/gpu_mode/prompt.py (TRIMUL_PROMPT, lines 1-184).
# ---------------------------------------------------------------------------

_TRIMUL_BASE_PROMPT = r'''You are an expert Triton engineer tasked with translating PyTorch code into highly optimized Triton kernel code.

You will be implementing a Triangle Multiplicative Update (TriMul) module that is a core operation
for AlphaFold3, Chai, Protenix, and other protein structure prediction models in BioML.

The TriMul operator operates over a 4D tensor of shape [B, N, N, C].

Your task:
- Implement the "outgoing" version of the TriMul operator from the AlphaFold3 paper.
- You will not have to compute or store gradients for this version. You will only need to implement the forward pass.

Your function should be defined as 'custom_kernel' with the following signature:
Input:
- `data`: Tuple of (input: torch.Tensor, weights: Dict[str, torch.Tensor], config: Dict)
    - input: Input tensor of shape [bs, seq_len, seq_len, dim]
    - mask: Mask tensor of shape [bs, seq_len, seq_len]
    - weights: Dictionary containing model weights
    - config: Dictionary containing model configuration parameters

Output:
- output: Processed tensor [bs, seq_len, seq_len, dim]

**Problem Constraints:**
- B ∈ {1,2}, N ∈ {128,256,512,1024}, c ∈ {128}, c_z ∈ {128,384,768}
- The input distribution will be sampled from a standard Normal distribution, or a heavy-tailed Cauchy distribution (gamma = 2).
- There will either be no mask, or a randomly sampled mask over the inputs.

**Remarks.** So why is this problem so annoying? Because you have to choose whether to load / deal with either the channel dimensions c,c_z that the LayerNorms require (otherwise you have to do a synchronize to compute the statistics like mean / variance) or the sequence dimension N.
The sequence dimension is particularly annoying because it's quite large, but also because we compute pair-wise operations at the last operation that sum over another sequence dimension (this is N^3!).
However, I really like this kernel because it only consists of "simple" operations, and is really easy to understand. It is a true test of "fusions" that torch.compile() doesn't do that well.

Here is a pytorch implementation of the TriMul module. You will want to implement a kernel for the operations in the forward call:

```python
import torch
from torch import nn, einsum
import math

# Reference code in PyTorch
class TriMul(nn.Module):
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        self.norm = nn.LayerNorm(dim)

        self.left_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.right_proj = nn.Linear(dim, hidden_dim, bias=False)

        self.left_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.right_gate = nn.Linear(dim, hidden_dim, bias=False)
        self.out_gate = nn.Linear(dim, hidden_dim, bias=False)

        self.to_out_norm = nn.LayerNorm(hidden_dim)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x: [bs, seq_len, seq_len, dim]
        mask: [bs, seq_len, seq_len]

        Returns:
            output: [bs, seq_len, seq_len, dim]
        """
        batch_size, seq_len, _, dim = x.shape

        x = self.norm(x)

        left = self.left_proj(x)
        right = self.right_proj(x)

        mask = mask.unsqueeze(-1)
        left = left * mask
        right = right * mask

        left_gate = self.left_gate(x).sigmoid()
        right_gate = self.right_gate(x).sigmoid()
        out_gate = self.out_gate(x).sigmoid()

        left = left * left_gate
        right = right * right_gate

        out = einsum('... i k d, ... j k d -> ... i j d', left, right)
        # This einsum is the same as the following:
        # out = torch.zeros(batch_size, seq_len, seq_len, dim, device=x.device)

        # # Compute using nested loops
        # for b in range(batch_size):
        #     for i in range(seq_len):
        #         for j in range(seq_len):
        #             # Compute each output element
        #             for k in range(seq_len):
        #                 out[b, i, j] += left[b, i, k, :] * right[b, j, k, :]

        out = self.to_out_norm(out)
        out = out * out_gate
        return self.to_out(out)
```

Here is some example skeleton code of the entrypoint function you will create:
```python
def custom_kernel(data)
    input_tensor, mask, weights, config = data
    dim, hidden_dim = config["dim"], config["hidden_dim"]

    # Access the given weights of the model
    norm_weight = weights["norm.weight"]
    norm_bias = weights["norm.bias"]
    left_proj_weight = weights["left_proj.weight"]
    right_proj_weight = weights["right_proj.weight"]
    left_gate_weight = weights["left_gate.weight"]
    right_gate_weight = weights["right_gate.weight"]
    out_gate_weight = weights["out_gate.weight"]
    to_out_norm_weight = weights["to_out_norm.weight"]
    to_out_norm_bias = weights["to_out_norm.bias"]
    to_out_weight = weights["to_out.weight"]

    # Perform TriMul

    return out
```

To help you understand which triton version we are using, here is some example triton code for an unrelated task:
```python
import triton
import triton.language as tl

@triton.jit
def matmul_persistent_ws_kernel(
   a_ptr, b_ptr, c_ptr, M, N, K,
   stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
   pid = tl.program_id(axis=0) # async_task 0, 1, 2
   num_pid_m = tl.cdiv(M, BLOCK_M) # async_task 0, 1, 2
   num_pid_n = tl.cdiv(N, BLOCK_N) # async_task 0, 1, 2
   pid_m = pid // num_pid_m # async_task 0, 1, 2
   pid_n = pid % num_pid_n # async_task 0, 1, 2
   offs_m_1 = pid_m * BLOCK_M + tl.arange(0, BLOCK_M // 2) # async_task 0, 1, 2
   offs_m_2 = pid_m * BLOCK_M + tl.arange(BLOCK_M // 2, BLOCK_M) # async_task 0, 1, 2
   offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_N) # async_task 0, 1, 2
   offs_k = tl.arange(0, BLOCK_K) # async_task 0
   a_ptrs_1 = a_ptr + (offs_m_1[:, None] * stride_am + offs_k[None, :] * stride_ak) # async_task 0
   a_ptrs_2 = a_ptr + (offs_m_2[:, None] * stride_am + offs_k[None, :] * stride_ak) # async_task 0
   b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn) # async_task 0
   acc_1 = tl.zeros((BLOCK_M // 2, BLOCK_N), dtype=tl.float32) # async_task 1
   acc_1 = tl.zeros((BLOCK_M // 2, BLOCK_N), dtype=tl.float32) # async_task 2
   for k in range(0, tl.cdiv(K, BLOCK_K)): # async_task 0, 1, 2
       a_1 = tl.load(a_ptrs_1)   # async_task 0
       a_2 = tl.load(a_ptrs_2)   # async_task 0
       b = tl.load(b_ptrs)   # async_task 0
       acc_1 += tl.dot(a_1, b)   # async_task 1
       acc_2 += tl.dot(a_2, b)   # async_task 2
       a_ptrs_1 += BLOCK_K * stride_ak # async_task 0
       a_ptrs_2 += BLOCK_K * stride_ak # async_task 0
       b_ptrs += BLOCK_K * stride_bk # async_task 0
   c_1 = acc_1.to(tl.float16) # async_task 1
   c_2 = acc_2.to(tl.float16) # async_task 2
   c_ptrs_1 = c_ptr_1 + stride_cm * offs_m_1[:, None] + stride_cn * offs_n[None, :] # async_task 1
   c_ptrs_2 = c_ptr_2 + stride_cm * offs_m_2[:, None] + stride_cn * offs_n[None, :] # async_task 2
   tl.store(c_ptrs_1, c_1) # async_task 1
   tl.store(c_ptrs_2, c_2) # async_task 2
```

A few general triton tips:
- tl.arange only takes in constexpr arguments (static or tl.constexpr)
- You cannot use continue in your kernel code
- tl.dot can only take in two input tensors
- There is no tl.mean

Here are the different configs that your kernel will be tested on ("nomask" sets whether there will be no mask, or a randomly sampled mask over the inputs):

Test Cases for correctness and runtime (optimize runtime for these):
  - {"seqlen": 256, "bs": 2, "dim": 128, "hidden_dim": 128, "nomask": True, "distribution": "normal"}
  - {"seqlen": 768, "bs": 1, "dim": 128, "hidden_dim": 128, "nomask": True, "distribution": "cauchy"}
  - {"seqlen": 256, "bs": 2, "dim": 384, "hidden_dim": 128, "nomask": False, "distribution": "normal"}
  - {"seqlen": 512, "bs": 1, "dim": 128, "hidden_dim": 128, "nomask": True, "distribution": "normal"}
  - {"seqlen": 1024, "bs": 1, "dim": 128, "hidden_dim": 128, "nomask": True, "distribution": "cauchy"}
  - {"seqlen": 768, "bs": 1, "dim": 384, "hidden_dim": 128, "nomask": False, "distribution": "normal"}
  - {"seqlen": 1024, "bs": 1, "dim": 384, "hidden_dim": 128, "nomask": True, "distribution": "normal"}
'''


# ---------------------------------------------------------------------------
# Rules block — appended after _TRIMUL_BASE_PROMPT, parameterised on the
# target GPU and Triton version. Mirrors the rules block in
# ttt-discover/examples/gpu_mode/env.py:199-207, with `gpu_name` and
# `triton_version` lifted to be runtime-configurable since we run on
# multiple GPUs (TTT hardcodes H100).
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
    return _TRIMUL_BASE_PROMPT.rstrip() + "\n\n" + rules


# ---------------------------------------------------------------------------
# Code extraction
# ---------------------------------------------------------------------------

# Mirrors ttt-discover/ttt_discover/tinker_utils/dataset_builder.py:139-173.
# Picks the LAST python block — required because the rules instruct the model
# to put its final code in one trailing python block, but reasoning models
# often emit one or more drafts above that final block.
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

# Per-feedback-kind truncation budgets. Tracebacks tail-truncate (the deepest
# frame is at the bottom and is usually the actionable signal); everything
# else head-truncates (first error line is usually the actionable signal).
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


def format_trimul_feedback_mutation_prompt(
    *,
    base_prompt: str,
    previous_kernel_code: str,
    feedback: TriMulKernelExecutionFeedback,
) -> str:
    """Build a mutation prompt from a base prompt and prior execution feedback.

    ``feedback`` must be one of the four concrete TriMul feedback types.
    Callers should pass the base prompt directly (zero-shot) for
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
                f"  seqlen={case.seqlen}, bs={case.bs}, dim={case.dim}, "
                f"hiddendim={case.hiddendim}, nomask={case.nomask}, "
                f"dist={case.distribution}: "
                f"{case.speedup:.3f}x "
                f"(ref: {ref_us:.1f}\u03bcs, candidate: {candidate_us:.1f}\u03bcs)\n"
            )
        prompt += (
            "\nPlease rewrite the entire kernel to be as fast as possible. "
            "Focus on the slowest configurations listed above."
        )

    return prompt


# ---------------------------------------------------------------------------
# Invocation record
# ---------------------------------------------------------------------------


class TriMulFeedbackMutationRecord(BaseModel, frozen=True):
    """Invocation record for one ``generate_mutations`` call.

    One call issues ``num_mutations_requested`` independent async LLM
    requests (each with implicit ``n=1``). ``num_llm_requests`` equals
    ``num_mutations_requested`` — every requested candidate corresponds
    to exactly one outbound call. ``num_mutations_produced`` may be
    smaller if some calls failed or their responses had no extractable
    code block.
    """

    kind: Literal["trimul_feedback_mutation"] = "trimul_feedback_mutation"
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


class TriMulFeedbackMutationProvider:
    """Generates TriMul kernel mutations conditioned on execution feedback.

    Implements ``MutationProvider[TriMulObservation]``.

    Unlike the KernelBench provider there is no ``reference_kernel_code``
    constructor argument — the TriMul task is fixed so the base prompt is
    a module-level constant.

    Parallelism is bounded by ``max_llm_concurrency``. Each outbound
    request inherits ``num_retries`` and ``request_timeout_s`` from
    litellm, so transient rate-limit and timeout failures are absorbed
    before they ever reach the provider.
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
        """
        ``max_tokens`` is the per-call output-token cap forwarded to
        ``litellm.acompletion`` when set. Leave as ``None`` (the default)
        for Gemini and other Google models — Gemini's generous default
        output budget is exactly what we want, and passing an explicit
        cap has historically interacted poorly with Gemini's thinking
        budget. Set explicitly (e.g. ``max_tokens=32000``) for gpt-oss
        via Together AI, where the provider default truncates at 4096
        and gpt-oss's analysis-channel reasoning alone consumes most of
        that before a final kernel is emitted.
        """
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
        evaluation: Evaluation[TriMulObservation],
    ) -> list[str]:
        feedback = evaluation.observation.feedback
        if isinstance(feedback, InfrastructureFailureFeedback):
            prompt = self._base_prompt
        else:
            prompt = format_trimul_feedback_mutation_prompt(
                base_prompt=self._base_prompt,
                previous_kernel_code=program_code,
                feedback=feedback,
            )

        parent_sha = code_sha256(program_code)
        logger.info(
            "TriMul mutation phase starting: parent_sha={parent}, n={n}, concurrency={c}",
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
            "TriMul mutation phase done: parent_sha={parent}, produced={produced}/{n}, "
            "elapsed={elapsed:.1f}s, in_tok={inp}, out_tok={out}",
            parent=parent_sha[:8],
            produced=len(codes),
            n=num_mutations,
            elapsed=wall_clock_seconds,
            inp=input_tokens,
            out=output_tokens,
        )

        if self._invocation_sink is not None:
            self._invocation_sink.record(
                TriMulFeedbackMutationRecord(
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
        """Fan out ``n`` independent ``acompletion(..., n=1)`` calls.

        Concurrency is bounded by ``self._max_llm_concurrency``. Each
        call's failure is logged and swallowed — the caller treats missing
        results as a shortfall rather than an error (PUCT handles fewer
        children than requested gracefully).

        Returns ``(codes, total_input_tokens, total_output_tokens)``.
        Token totals sum over all calls (including failed ones where
        token counts are 0).
        """
        semaphore = asyncio.Semaphore(self._max_llm_concurrency)

        async def _single_call(index: int) -> tuple[str | None, int, int]:
            async with semaphore:
                logger.info("LLM call {index}/{n}: starting", index=index, n=n)
                call_start_s = time.perf_counter()
                try:
                    # Only forward ``max_tokens`` when the caller set one —
                    # Gemini runs historically rely on litellm's default
                    # (unbounded) behaviour here.
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
                        "LLM call {index}/{n}: no code block extracted (elapsed={elapsed:.1f}s, in_tok={inp}, out_tok={out}).",
                        index=index,
                        n=n,
                        elapsed=elapsed,
                        inp=input_tokens,
                        out=output_tokens,
                    )
                    return None, input_tokens, output_tokens
                logger.info(
                    "LLM call {index}/{n}: done (elapsed={elapsed:.1f}s, in_tok={inp}, out_tok={out}, code_chars={chars})",
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


implements(MutationProvider[TriMulObservation])(TriMulFeedbackMutationProvider)
