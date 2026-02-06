from __future__ import annotations

from typing import Literal

from kernelbench.prompt_constructor_toml import get_prompt_for_backend
from kernelbench.utils import extract_first_code
from litellm import completion
from loguru import logger
import time

from arid_badger.typing_utils import implements
from .domain import (
    CompileFailedFeedback,
    IncorrectFeedback,
    KernelExecutionFeedback,
    MutationContext,
    MutationFunction,
    MutatedKernel,
    RuntimeErrorFeedback,
    SuccessFeedback,
)


def _truncate(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n[truncated]"


def format_feedback_mutation_prompt(
    *,
    base_prompt: str,
    previous_kernel_code: str,
    feedback: KernelExecutionFeedback,
    max_chars: int = 2000,
) -> str:
    if not base_prompt:
        raise ValueError("base_prompt must be non-empty.")
    if not previous_kernel_code:
        raise ValueError("previous_kernel_code must be non-empty.")
    if feedback is None:
        raise ValueError("feedback is required for feedback-based mutation prompts.")

    prompt = base_prompt.rstrip() + "\n"
    prompt += "\nHere is your latest generation:\n"
    prompt += f"```\n{previous_kernel_code}\n```\n\n"
    prompt += (
        "Your generated architecture ModelNew and kernel were evaluated on GPU.\n\n"
    )
    prompt += "Here is the evaluation result:\n"

    if isinstance(feedback, CompileFailedFeedback):
        prompt += "Your kernel failed to compile.\n\n"
        prompt += "Compilation error name:\n"
        prompt += f"{_truncate(feedback.compilation_error_name, max_chars)}\n\n"
        prompt += "Compilation error details:\n"
        prompt += f"{_truncate(feedback.compilation_error, max_chars)}\n\n"
        prompt += "Please fix the errors and try again."
    elif isinstance(feedback, RuntimeErrorFeedback):
        prompt += "Your kernel failed to run due to a runtime error.\n\n"
        prompt += "Runtime error name:\n"
        prompt += f"{_truncate(feedback.runtime_error_name, max_chars)}\n\n"
        prompt += "Runtime error details:\n"
        prompt += f"{_truncate(feedback.runtime_error, max_chars)}\n\n"
        prompt += "Runtime error traceback:\n"
        prompt += f"{_truncate(feedback.runtime_error_traceback, max_chars)}\n\n"
        prompt += "Please fix the errors and try again."
    elif isinstance(feedback, IncorrectFeedback):
        prompt += (
            "Your kernel failed to produce the correct output compared to the "
            "reference architecture.\n\n"
        )
        prompt += "Correctness issue:\n"
        prompt += f"{_truncate(feedback.correctness_issue, max_chars)}\n\n"
        if feedback.max_difference:
            prompt += f"Max difference samples: {feedback.max_difference}\n"
        if feedback.avg_difference:
            prompt += f"Avg difference samples: {feedback.avg_difference}\n"
        prompt += "\nPlease regenerate ModelNew while fixing correctness issues."
    elif isinstance(feedback, SuccessFeedback):
        prompt += (
            "Your kernel executed successfully and produced the correct output.\n\n"
        )
        prompt += (
            "Wall clock times (microseconds): "
            f"runtime={feedback.runtime_us}, ref_runtime={feedback.ref_runtime_us}\n\n"
        )
        prompt += f"Speedup: {feedback.speedup:.4f}x\n\n"
        prompt += "Please rewrite the entire kernel to be as fast as possible."
    else:
        raise ValueError(f"Unknown feedback kind: {feedback}")

    return prompt


class KernelBenchExecutionFeedbackMutationFunction:
    """Generates mutated kernels using execution feedback."""

    def __init__(
        self,
        *,
        model_slug: str = "gemini/gemini-3-flash-preview",
        prompt_option: Literal["zero_shot", "one_shot", "few_shot"] = "one_shot",
    ) -> None:
        self._model_slug = model_slug
        self._prompt_option = prompt_option

    def __call__(self, context: MutationContext) -> MutatedKernel:
        if context.previous_evaluation is None:
            raise ValueError(
                "KernelBenchExecutionFeedbackMutationFunction requires "
                "context.previous_evaluation."
            )
        feedback = context.previous_evaluation.execution_feedback
        base_prompt = get_prompt_for_backend(
            ref_arch_src=context.reference_kernel_code,
            backend=context.backend,
            option=self._prompt_option,
            precision=context.precision,
        )
        prompt = format_feedback_mutation_prompt(
            base_prompt=base_prompt,
            previous_kernel_code=context.previous_kernel_code,
            feedback=feedback,
        )

        logger.info(
            "Feedback mutation request sent to LLM (model={model}, prompt_option={option}).",
            model=self._model_slug,
            option=self._prompt_option,
        )
        logger.debug(
            "Feedback mutation prompt length={length} chars.",
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
            "Feedback mutation response received from LLM (chars={length}, elapsed_s={elapsed_s:.2f}).",
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
            "Feedback mutation produced kernel code (chars={length}).",
            length=len(kernel_code),
        )
        return mutated


implements(MutationFunction)(KernelBenchExecutionFeedbackMutationFunction)
