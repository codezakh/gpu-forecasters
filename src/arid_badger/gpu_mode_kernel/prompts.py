"""Generic prompt assembly + code extraction for gpu-mode kernel mutations.

Generalizes ``arid_badger.hill_climbing.mutation_providers.<kernel>_feedback_mutation``:

- The rules block, code-block extraction regex, truncation helpers, and
  failure-arm formatters lift verbatim — they were byte-identical
  across the per-kernel mutation providers.
- The kernel-specific narrative (op description, reference source, test
  cases listing) lives on ``KernelPack.kernel_description_body`` —
  built at the pack's module-load time so the pack carries one ready
  string.
- The per-case success-path formatter is ``pack.case_speedup_format``;
  the surrounding "Per-case breakdown (slowest first):" wrapper is
  generic and stays here.
"""

from __future__ import annotations

import re
from typing import assert_never

from arid_badger.gpu_mode_kernel.core import (
    CaseSpeedupBase,
    CaseSpeedupT,
    CompileFailedFeedback,
    IncorrectFeedback,
    KernelExecutionFeedback,
    RuntimeErrorFeedback,
    SuccessFeedback,
)
from arid_badger.gpu_mode_kernel.kernel_pack import KernelPack, TestArgsT


# ---------------------------------------------------------------------------
# Rules block — appended to every base prompt.
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
    """Assemble the full base prompt: kernel description + rules block.

    The pack's ``kernel_description_body`` is responsible for everything
    up to (and including) the test cases listing — that lets each pack
    format its shape fields the way upstream gpu-mode/reference-kernels
    expects them.
    """
    rules = _RULES_TEMPLATE.format(gpu_name=gpu_name, triton_version=triton_version)
    return pack.kernel_description_body.rstrip() + "\n\n" + rules


# ---------------------------------------------------------------------------
# Code extraction — kernel-agnostic.
# ---------------------------------------------------------------------------

# Picks the LAST python block — the rules instruct the model to put its
# final code in one trailing block, but reasoning models often emit
# drafts above that final block.
_PYTHON_CODEBLOCK_RE = re.compile(
    r"```python\n(?!```)(.*?)(?:\n```)?(?=\n```|$)",
    re.DOTALL,
)


def extract_last_python_codeblock(text: str) -> str | None:
    matches = list(_PYTHON_CODEBLOCK_RE.finditer(text))
    if not matches:
        return None
    code = matches[-1].group(1).rstrip()
    return code or None


# ---------------------------------------------------------------------------
# Truncation budgets + helpers.
# ---------------------------------------------------------------------------

# Tracebacks tail-truncate (deepest frame is at the bottom — usually the
# actionable signal); everything else head-truncates.
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
# Feedback prompt assembly — generic over the pack's success arm.
# ---------------------------------------------------------------------------


def format_feedback_prompt[CaseSpeedupT: CaseSpeedupBase](
    *,
    base_prompt: str,
    previous_kernel_code: str,
    feedback: KernelExecutionFeedback[CaseSpeedupT],
) -> str:
    """Build a mutation prompt from a base prompt and prior execution feedback.

    ``feedback`` is one of the four in-band ``KernelExecutionFeedback``
    arms. Callers pass the base prompt directly (zero-shot) for
    ``InfrastructureFailureFeedback`` rather than going through this
    helper, since the harness fault carries no signal worth feeding to
    the LLM.
    """
    if not base_prompt:
        raise ValueError("base_prompt must be non-empty.")
    if not previous_kernel_code:
        raise ValueError("previous_kernel_code must be non-empty.")

    prompt = base_prompt.rstrip() + "\n"
    prompt += "\nHere is your latest implementation:\n"
    prompt += f"```python\n{previous_kernel_code}\n```\n\n"
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
