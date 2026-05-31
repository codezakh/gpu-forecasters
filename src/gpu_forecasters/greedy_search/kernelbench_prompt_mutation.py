from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from kernelbench.prompt_constructor_toml import get_prompt_for_backend
from kernelbench.utils import extract_first_code
from litellm import completion
from loguru import logger
import time
from pydantic import BaseModel

from gpu_forecasters.invocation_sink import InvocationSink, code_sha256
from gpu_forecasters.typing_utils import implements
from .domain import MutationContext, MutationFunction, MutatedKernel


class PromptMutationRecord(BaseModel, frozen=True):
    """Invocation record for a single KernelBench prompt mutation LLM call."""

    kind: Literal["prompt_mutation"] = "prompt_mutation"
    parent_code_sha256: str
    child_code_sha256: str
    model_slug: str
    input_tokens: int
    output_tokens: int
    wall_clock_seconds: float
    timestamp_utc: str


class KernelBenchPromptMutationFunction:
    """Generates mutated kernels using the default KernelBench prompt."""

    def __init__(
        self,
        *,
        model_slug: str = "gemini/gemini-3-flash-preview",
        prompt_option: Literal["zero_shot", "one_shot", "few_shot"] = "one_shot",
        invocation_sink: InvocationSink | None = None,
    ) -> None:
        self._model_slug = model_slug
        self._prompt_option = prompt_option
        self._invocation_sink = invocation_sink

    def _build_prompt(self, context: MutationContext) -> str:
        return get_prompt_for_backend(
            ref_arch_src=context.reference_kernel_code,
            backend=context.backend,
            option=self._prompt_option,
            precision=context.precision,
        )

    def __call__(self, context: MutationContext) -> MutatedKernel:
        prompt = self._build_prompt(context)
        logger.info(
            "Mutation request sent to LLM (model={model}, prompt_option={option}).",
            model=self._model_slug,
            option=self._prompt_option,
        )
        logger.debug(
            "Mutation prompt length={length} chars.",
            length=len(prompt),
        )
        start_time_s = time.perf_counter()
        response = completion(
            model=self._model_slug,
            messages=[{"role": "user", "content": prompt}],
            timeout=20.0,
        )
        elapsed_s = time.perf_counter() - start_time_s

        content = response.choices[  # pyright: ignore[reportAttributeAccessIssue]
            0
        ].message.content  # pyright: ignore[reportAttributeAccessIssue]
        if content is None:
            raise ValueError("LLM returned empty content")
        logger.info(
            "Mutation response received from LLM (chars={length}, elapsed_s={elapsed_s:.2f}).",
            length=len(content),
            elapsed_s=elapsed_s,
        )

        kernel_code = extract_first_code(content, code_language_types=["python"])
        if not kernel_code:
            logger.debug("LLM response preview:\n{preview}", preview=content[:500])
            raise ValueError(
                f"Could not extract code from LLM response. Response: {content[:500]}"
            )

        mutated = MutatedKernel(
            kernel_code=kernel_code,
            ancestor_ulid=context.previous_kernel_ulid,
        )
        logger.success(
            "Mutation produced kernel code (chars={length}).",
            length=len(kernel_code),
        )

        if self._invocation_sink is not None:
            raw_usage = response.usage  # pyright: ignore[reportAttributeAccessIssue]
            if raw_usage is not None:
                self._invocation_sink.record(
                    PromptMutationRecord(
                        parent_code_sha256=code_sha256(context.previous_kernel_code),
                        child_code_sha256=code_sha256(kernel_code),
                        model_slug=self._model_slug,
                        input_tokens=raw_usage.prompt_tokens,
                        output_tokens=raw_usage.completion_tokens,
                        wall_clock_seconds=elapsed_s,
                        timestamp_utc=datetime.now(timezone.utc).isoformat(),
                    )
                )

        return mutated


implements(MutationFunction)(KernelBenchPromptMutationFunction)
