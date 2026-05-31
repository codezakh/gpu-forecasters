"""Mutation-prompt assembly for the compound-observation search loop.

This module is intentionally a sibling, not a wrapper, of
``gpu_forecasters.gpu_mode_kernel.prompts``. It owns:

* a copy of the rules template and base-prompt assembly the
  gpu_mode_kernel mutation provider uses;
* a real-eval feedback formatter covering the four in-band
  ``KernelExecutionFeedback`` arms (compile / runtime / incorrect /
  success);
* a forecast feedback formatter covering the surrogate-forecast
  observation arm.

The duplication is deliberate. The compound mutation provider needs
to evolve its real-eval-arm prompt independently of the
gpu_mode_kernel one (e.g. to mention the surrogate context, to mark
forecast-derived parents differently when they appear in the
history, to add abstention-specific framing). Keeping this surface
local avoids the brittleness of a wrapping/delegation chain.

The closing rewrite instruction and the parent-code block are
identical across both arms so the mutator's output contract does
not change with the parent's observation type.
"""

from __future__ import annotations

from typing import assert_never

from gpu_forecasters.abstaining_evaluation.v1.observation import ForecastObservation
from gpu_forecasters.gpu_mode_kernel.core import (
    CaseSpeedupBase,
    CaseSpeedupT,
    CompileFailedFeedback,
    IncorrectFeedback,
    KernelExecutionFeedback,
    RuntimeErrorFeedback,
    SuccessFeedback,
)
from gpu_forecasters.gpu_mode_kernel.kernel_pack import KernelPack, TestArgsT
from gpu_forecasters.landscape_map.v2 import SUCCESS_BINS


# ---------------------------------------------------------------------------
# Rules block + base prompt — copied from gpu_mode_kernel.prompts at fork
# time and free to diverge.
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


def build_base_prompt(
    pack: KernelPack[TestArgsT, CaseSpeedupT],
    *,
    gpu_name: str,
    triton_version: str,
) -> str:
    """Assemble the full base prompt: kernel description + rules block."""
    rules = _RULES_TEMPLATE.format(gpu_name=gpu_name, triton_version=triton_version)
    return pack.kernel_description_body.rstrip() + "\n\n" + rules


# ---------------------------------------------------------------------------
# Truncation budgets — copied at fork time.
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


# ---------------------------------------------------------------------------
# Real-eval-arm feedback formatter (four in-band arms).
# ---------------------------------------------------------------------------


def format_real_eval_feedback_prompt[T: CaseSpeedupBase](
    *,
    base_prompt: str,
    parent_code: str,
    feedback: KernelExecutionFeedback[T],
) -> str:
    """Build a mutation prompt from a base prompt and a real-eval feedback.

    Mirrors ``gpu_mode_kernel.prompts.format_feedback_prompt`` at fork
    time. Callers handle ``InfrastructureFailureFeedback`` by emitting
    the base prompt directly (no execution feedback to render).
    """
    if not base_prompt:
        raise ValueError("base_prompt must be non-empty.")
    if not parent_code:
        raise ValueError("parent_code must be non-empty.")

    prompt = base_prompt.rstrip() + "\n"
    prompt += "\nHere is your latest implementation:\n"
    prompt += f"```python\n{parent_code}\n```\n\n"
    prompt += "Your custom_kernel was evaluated on GPU.\n\nHere is the evaluation result:\n"

    match feedback:
        case CompileFailedFeedback():
            prompt += "Your kernel failed to compile.\n\n"
            prompt += "Compilation error:\n"
            prompt += (
                _truncate_head(
                    feedback.compilation_error, _MAX_COMPILATION_ERROR_CHARS
                )
                + "\n\n"
            )
            prompt += "Please fix the errors and try again."
        case RuntimeErrorFeedback():
            prompt += "Your kernel raised an exception at runtime.\n\n"
            prompt += f"Error type: {feedback.runtime_error_name}\n\n"
            prompt += "Error message:\n"
            prompt += (
                _truncate_head(feedback.runtime_error, _MAX_RUNTIME_ERROR_CHARS)
                + "\n\n"
            )
            prompt += "Traceback:\n"
            prompt += (
                _truncate_tail(feedback.traceback, _MAX_TRACEBACK_CHARS) + "\n\n"
            )
            prompt += "Please fix the errors and try again."
        case IncorrectFeedback():
            prompt += "Your kernel produced incorrect output compared to the reference.\n\n"
            prompt += "Correctness issue:\n"
            prompt += (
                _truncate_head(feedback.error_message, _MAX_INCORRECT_ERROR_CHARS)
                + "\n\n"
            )
            prompt += "Please fix the correctness issues and try again."
        case SuccessFeedback():
            prompt += "You are iteratively optimizing runtime (microseconds).\n\n"
            prompt += (
                f"Your kernel is correct. "
                f"Aggregated speedup: {feedback.aggregated_speedup:.3f}x "
                f"(aggregation method: {feedback.aggregation_method}).\n\n"
            )
            sorted_cases = sorted(
                feedback.per_case_speedups, key=lambda c: c.speedup
            )
            prompt += "Per-case breakdown (slowest first):\n"
            for case in sorted_cases:
                prompt += f"  {case.format_for_prompt()}\n"
            prompt += (
                "\nPlease rewrite the entire kernel to be as fast as possible. "
                "Focus on the slowest configurations listed above."
            )
        case _:
            assert_never(feedback)

    return prompt


# ---------------------------------------------------------------------------
# Forecast-arm feedback formatter.
# ---------------------------------------------------------------------------


def format_forecast_feedback_prompt(
    *,
    base_prompt: str,
    parent_code: str,
    forecast: ForecastObservation,
) -> str:
    """Build a mutation prompt for a parent whose evaluation is a
    surrogate forecast rather than a GPU run.

    The scaffolding (rules block, parent-code block, closing rewrite
    instruction) is the same shape as the real-eval prompt so the
    mutator's output contract does not depend on the parent's
    observation arm. Only the middle "evaluation result" block changes:
    we explicitly mark the result as a forecast, render the predicted
    bin, the full bin distribution, the surrogate's reasoning, and a
    closing reminder that the forecast is a prediction, not a
    measurement.
    """
    if not base_prompt:
        raise ValueError("base_prompt must be non-empty.")
    if not parent_code:
        raise ValueError("parent_code must be non-empty.")

    estimate = forecast.estimate

    prompt = base_prompt.rstrip() + "\n"
    prompt += "\nHere is your latest implementation:\n"
    prompt += f"```python\n{parent_code}\n```\n\n"
    prompt += (
        "Your custom_kernel was REVIEWED BY A SURROGATE PREDICTOR "
        "(no GPU run).\n\n"
        "Forecast (this is a prediction, not a measurement):\n"
    )
    prompt += f"  Predicted bin: {estimate.predicted_bin.label}\n"
    prompt += (
        f"  Expected speedup under the distribution: "
        f"{forecast.expected_speedup:.3f}x\n"
    )
    prompt += "  Distribution over speedup bins:\n"
    for bin_ in SUCCESS_BINS:
        p = estimate.bin_probabilities[bin_]
        prompt += f"    bin {int(bin_)} {bin_.label}: {p:.3f}\n"
    prompt += f"  Surrogate reasoning:\n    {estimate.reasoning}\n\n"
    prompt += (
        "The forecast above reflects the surrogate's prediction, not "
        "measured runtime. Please rewrite the entire kernel to be as "
        "fast as possible."
    )
    return prompt


__all__ = [
    "build_base_prompt",
    "format_forecast_feedback_prompt",
    "format_real_eval_feedback_prompt",
]
