"""End-to-end check that the TriMul prompt we feed GPT-OSS is compatible
with our scoring path.

Sends the exact prompt produced by ``GpuModeEnv.get_question()`` (what the
RL rollouts will see on cold start) to Together AI via LiteLLM for both
``gpt-oss-20b`` and ``gpt-oss-120b``. For every completion we run the
generated output through the same code-block extractor the training loop
uses and then score it with ``GpuModeRewardEvaluator`` on Modal.

What this test asserts:
- At least one of ``N_SAMPLES`` completions per model produces a
  parseable ``` ```python ``` ``` block that our extractor recovers —
  i.e. the prompt shape is compatible with the model's response format.
- ``GpuModeRewardEvaluator.get_reward`` returns a well-formed reward dict
  for every extracted kernel (no infra failures short of actual scoring).

What this test deliberately does NOT assert: that any kernel is
correct. Correctness on cold-start uprompted gpt-oss-20b is expected to
be zero per the TTT-Discover paper; we surface the observed counts in
the test output for eyeballing, but we don't fail on them.

Run:

    uv run --env-file .env pytest -m "modal and integration" \
        tests/ttt_discover/test_prompt_compat_integration.py -v -s
"""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import litellm
import pytest

from gpu_forecasters.ttt_discover.v1.examples.gpu_mode.env import (
    GpuModeEnv,
    GpuModeRewardEvaluator,
)
from gpu_forecasters.ttt_discover.v1.tinker_utils.dataset_builder import (
    last_codeblock_postprocess,
)


pytestmark = [pytest.mark.modal, pytest.mark.integration]


# Together AI model slugs for LiteLLM. Together hosts the gpt-oss series
# under the upstream ``openai/gpt-oss-*`` names.
_GPT_OSS_20B = "together_ai/openai/gpt-oss-20b"
_GPT_OSS_120B = "together_ai/openai/gpt-oss-120b"

# Per-model concurrent samples. 4 is enough to distinguish "prompt is
# broken, 0/4 parse" from "prompt works, 4/4 parse" without burning a
# lot of tokens.
N_SAMPLES = 4

# Triton kernels + explanation easily blow through small token budgets.
MAX_TOKENS = 8192
REQUEST_TIMEOUT_S = 600.0


def _build_trimul_question() -> str:
    """Reproduce ``GpuModeEnv.get_question()`` on the RL cold-start state
    without running the full ``Environment.__init__`` (which pulls in
    renderer/sampler/config machinery irrelevant to prompt construction)."""
    env = GpuModeEnv.__new__(GpuModeEnv)
    env.initial_state = GpuModeEnv.create_initial_state("trimul")
    return env.get_question()


def _one_completion(model: str, prompt: str, idx: int) -> tuple[int, str]:
    start = time.perf_counter()
    response = litellm.completion(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
        max_tokens=MAX_TOKENS,
        timeout=REQUEST_TIMEOUT_S,
        num_retries=2,
    )
    content = response.choices[0].message.content or ""  # pyright: ignore[reportAttributeAccessIssue]
    elapsed = time.perf_counter() - start
    print(
        f"[{model} #{idx}] completion in {elapsed:.1f}s, "
        f"{len(content)} chars"
    )
    return idx, content


def _require_together_api_key() -> None:
    if not (os.environ.get("TOGETHER_API_KEY") or os.environ.get("TOGETHER_AI_API_KEY")):
        pytest.skip("TOGETHER_API_KEY not set; skipping Together AI test")


@pytest.mark.parametrize("model_slug", [_GPT_OSS_20B, _GPT_OSS_120B])
def test_trimul_prompt_is_compatible_with_gpt_oss(model_slug: str) -> None:
    _require_together_api_key()

    prompt = _build_trimul_question()
    # Sanity-check the prompt we're actually sending — if the TriMul
    # payload was accidentally trimmed we'd see a short question.
    assert "TriMul" in prompt or "Triangle Multiplicative" in prompt, prompt[:500]
    assert "custom_kernel" in prompt

    # Fan out N concurrent samples. We use threads because LiteLLM's
    # sync `completion` releases the GIL during the HTTP wait.
    contents: list[str] = [""] * N_SAMPLES
    with ThreadPoolExecutor(max_workers=N_SAMPLES) as pool:
        futures = [
            pool.submit(_one_completion, model_slug, prompt, i)
            for i in range(N_SAMPLES)
        ]
        for fut in as_completed(futures):
            idx, content = fut.result()
            contents[idx] = content

    # Extract code using the same post-processor the training loop uses
    # (see dataset_builder.Environment.step). GpuModeEnv sets
    # keep_separators=False, so the returned code is raw python source.
    codes: list[str] = []
    for i, content in enumerate(contents):
        code = last_codeblock_postprocess(
            content,
            codeblock_seps=["python"],
            keep_separators=False,
            last_response_strict=True,
        )
        if code:
            codes.append(code)
        print(
            f"[{model_slug} #{i}] extracted code: "
            f"{'yes' if code else 'NO'} ({len(code)} chars)"
        )

    assert codes, (
        f"No sample out of {N_SAMPLES} from {model_slug} produced a "
        f"parseable ```python block. First 500 chars of first completion: "
        f"{contents[0][:500]!r}"
    )

    # Score each extracted kernel on Modal through the evaluator used in
    # RL rollouts. The evaluator opens a process-global Modal session
    # lazily on first call.
    evaluator = GpuModeRewardEvaluator(problem_type="trimul", log_dir=None)
    initial_state = GpuModeEnv.create_initial_state("trimul")

    num_format_ok = len(codes)
    num_triton = 0
    num_correct = 0
    rewards: list[float] = []
    best_raw_score_us: float | None = None

    for i, code in enumerate(codes):
        if "@triton.jit" in code:
            num_triton += 1
        result = evaluator.get_reward(code, initial_state)
        # Contract: training loop depends on these keys.
        for key in ("reward", "msg", "correctness", "raw_score"):
            assert key in result, f"missing key {key!r} in reward dict: {result}"
        correctness = float(result["correctness"])
        reward = float(result["reward"])
        rewards.append(reward)
        if correctness > 0:
            num_correct += 1
            raw_us = float(result["raw_score"])
            if best_raw_score_us is None or raw_us < best_raw_score_us:
                best_raw_score_us = raw_us
        print(
            f"[{model_slug} #{i}] correctness={correctness} reward={reward:.4f} "
            f"msg={result['msg'][:180]!r}"
        )

    print(
        f"[{model_slug}] summary: format_ok={num_format_ok}/{N_SAMPLES} "
        f"triton_jit={num_triton}/{N_SAMPLES} correct={num_correct}/{N_SAMPLES} "
        f"best_raw_score_us={best_raw_score_us}"
    )

    # Informational: we do not fail on zero correctness (cold-start is
    # expected to be near-zero per TTT-Discover paper). But we DO fail
    # if no sample even looks like triton — that would mean the prompt
    # isn't steering the model toward the intended output shape.
    assert num_triton > 0, (
        f"None of {N_SAMPLES} {model_slug} samples contained @triton.jit. "
        f"Prompt may not be steering the model to write a triton kernel."
    )
