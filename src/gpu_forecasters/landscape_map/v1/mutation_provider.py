"""Mutation provider that uses landscape map model feedback to guide LLM mutations."""

from __future__ import annotations

import asyncio
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import litellm
from jinja2 import Environment, FileSystemLoader
from loguru import logger
from pydantic import BaseModel

from gpu_forecasters.hill_climbing.domain import Evaluation, MutationProvider
from gpu_forecasters.invocation_sink import InvocationSink, code_sha256
from gpu_forecasters.landscape_map.v1.domain import KernelRuntimeEstimate, SpeedupBin
from gpu_forecasters.typing_utils import implements

from kernelbench.prompt_constructor_toml import get_prompt_for_backend
from kernelbench.utils import extract_first_code


class LandscapeMapMutationRecord(BaseModel, frozen=True):
    """Invocation record for one generate_mutations call (aggregates N async LLM calls)."""

    kind: Literal["landscape_map_mutation"] = "landscape_map_mutation"
    parent_code_sha256: str
    child_code_sha256s: list[str]
    model_slug: str
    total_input_tokens: int
    total_output_tokens: int
    num_mutations_requested: int
    num_mutations_produced: int
    wall_clock_seconds: float
    timestamp_utc: str


_PROMPTS_DIR = Path(__file__).parent / "prompts"
_JINJA_ENV = Environment(
    loader=FileSystemLoader(str(_PROMPTS_DIR)),
    autoescape=False,
)


def format_landscape_map_feedback_prompt(
    *,
    base_prompt: str,
    previous_kernel_code: str,
    estimate: KernelRuntimeEstimate,
) -> str:
    """Format a mutation prompt using the landscape map model's assessment."""
    template = _JINJA_ENV.get_template("mutation_feedback.j2")
    return template.render(
        base_prompt=base_prompt.rstrip(),
        previous_kernel_code=previous_kernel_code,
        predicted_bin_label=estimate.predicted_bin.label,
        reasoning=estimate.reasoning,
    )


class LandscapeMapModelMutationProvider:
    """Generates mutations conditioned on landscape map model feedback.

    When the landscape map model predicts failure (bin 0), falls back to the
    zero-shot base prompt. Otherwise, includes the model's reasoning and
    predicted bin label to guide the mutation LLM toward better kernels.
    """

    def __init__(
        self,
        *,
        reference_kernel_code: str,
        model_slug: str,
        backend: str = "cuda",
        precision: str = "fp32",
        max_llm_concurrency: int = 8,
        invocation_sink: InvocationSink | None = None,
    ) -> None:
        self._base_prompt = get_prompt_for_backend(
            ref_arch_src=reference_kernel_code,
            backend=backend,
            option="zero_shot",
            precision=precision,
        )
        self._model_slug = model_slug
        self._max_llm_concurrency = max_llm_concurrency
        self._invocation_sink = invocation_sink

    def generate_mutations(
        self,
        program_code: str,
        num_mutations: int,
        evaluation: Evaluation[KernelRuntimeEstimate],
    ) -> list[str]:
        estimate = evaluation.observation
        if estimate.predicted_bin == SpeedupBin.FAILURE:
            prompt = self._base_prompt
        else:
            prompt = format_landscape_map_feedback_prompt(
                base_prompt=self._base_prompt,
                previous_kernel_code=program_code,
                estimate=estimate,
            )
        start_time_s = time.perf_counter()
        codes, total_input_tokens, total_output_tokens = asyncio.run(
            self._generate_async(prompt, num_mutations)
        )
        wall_clock_seconds = time.perf_counter() - start_time_s

        if self._invocation_sink is not None:
            self._invocation_sink.record(
                LandscapeMapMutationRecord(
                    parent_code_sha256=code_sha256(program_code),
                    child_code_sha256s=[code_sha256(c) for c in codes],
                    model_slug=self._model_slug,
                    total_input_tokens=total_input_tokens,
                    total_output_tokens=total_output_tokens,
                    num_mutations_requested=num_mutations,
                    num_mutations_produced=len(codes),
                    wall_clock_seconds=wall_clock_seconds,
                    timestamp_utc=datetime.now(timezone.utc).isoformat(),
                )
            )

        return codes

    async def _generate_async(
        self, prompt: str, n: int
    ) -> tuple[list[str], int, int]:
        """Fire n async LLM calls with semaphore-based concurrency control.

        Returns a tuple of (codes, total_input_tokens, total_output_tokens).
        """
        semaphore = asyncio.Semaphore(self._max_llm_concurrency)

        async def _single_call(index: int) -> tuple[str | None, int, int]:
            async with semaphore:
                try:
                    response = await litellm.acompletion(
                        model=self._model_slug,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=1.0,
                    )
                    raw_usage = response.usage
                    input_tokens = raw_usage.prompt_tokens if raw_usage is not None else 0
                    output_tokens = raw_usage.completion_tokens if raw_usage is not None else 0
                    content = response.choices[0].message.content
                    code = extract_first_code(content, code_language_types=["python"])
                    if not code:
                        logger.warning(
                            "LLM call {index}: no code block extracted.",
                            index=index,
                        )
                        return None, input_tokens, output_tokens
                    return code, input_tokens, output_tokens
                except Exception:
                    logger.warning(
                        "LLM call {index} failed:\n{tb}",
                        index=index,
                        tb=traceback.format_exc(),
                    )
                    return None, 0, 0

        results = await asyncio.gather(*[_single_call(i) for i in range(n)])
        codes = [code for code, _, _ in results if code is not None]
        total_input_tokens = sum(inp for _, inp, _ in results)
        total_output_tokens = sum(out for _, _, out in results)
        return codes, total_input_tokens, total_output_tokens


implements(MutationProvider)(LandscapeMapModelMutationProvider)
