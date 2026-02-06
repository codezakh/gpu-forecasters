from __future__ import annotations

from typing import Literal

from kernelbench.prompt_constructor_toml import get_prompt_for_backend
from kernelbench.utils import extract_first_code
from litellm import completion
from loguru import logger
import time

from arid_badger.typing_utils import implements
from .domain import MutationContext, MutationFunction, MutatedKernel


class KernelBenchPromptMutationFunction:
    """Generates mutated kernels using the default KernelBench prompt."""

    def __init__(
        self,
        *,
        model_slug: str = "gemini/gemini-3-flash-preview",
        prompt_option: Literal["zero_shot", "one_shot", "few_shot"] = "one_shot",
    ) -> None:
        self._model_slug = model_slug
        self._prompt_option = prompt_option

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
        return mutated


implements(MutationFunction)(KernelBenchPromptMutationFunction)
