"""Feedback-prompt renderer for TriMul.

Mirrors ``gpu_forecasters.hill_climbing.mutation_providers.trimul_feedback
_mutation.format_trimul_feedback_mutation_prompt``, but lifted into a
class so the truncation budgets are constructor knobs and the cold-start
path (no parent → no feedback section, just an empty string so the env
only emits the task prompt) is represented explicitly.

The renderer produces only the *feedback* block: the block that v1
dropped on the floor at the reward-dict boundary. The task prompt (base
instructions + rules block) is produced separately by a ``TaskPromptRenderer``;
the environment concatenates them.
"""

from __future__ import annotations

from gpu_forecasters.trimul.core import (
    CompileFailedFeedback,
    IncorrectFeedback,
    InfrastructureFailureFeedback,
    RuntimeErrorFeedback,
    SuccessFeedback,
)
from gpu_forecasters.ttt_discover.v2.domain.context import FeedbackPromptContext
from gpu_forecasters.ttt_discover.v2.domain.outcome import ParseFailureFeedback
from gpu_forecasters.ttt_discover.v2.interfaces.renderer import FeedbackPromptRenderer
from gpu_forecasters.typing_utils import implements


def _truncate_head(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return f"{text[:max_chars]}\n[truncated]"


def _truncate_tail(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return f"[truncated]\n...{text[-max_chars:]}"


class TriMulFeedbackPromptRenderer:
    _max_compilation_error_chars: int
    _max_runtime_error_chars: int
    _max_traceback_chars: int
    _max_incorrect_error_chars: int

    def __init__(
        self,
        *,
        max_compilation_error_chars: int = 2000,
        max_runtime_error_chars: int = 1000,
        max_traceback_chars: int = 3000,
        max_incorrect_error_chars: int = 2000,
    ) -> None:
        self._max_compilation_error_chars = max_compilation_error_chars
        self._max_runtime_error_chars = max_runtime_error_chars
        self._max_traceback_chars = max_traceback_chars
        self._max_incorrect_error_chars = max_incorrect_error_chars

    def render(self, ctx: FeedbackPromptContext) -> str:
        parent = ctx.parent
        if parent is None or parent.outcome is None or not parent.code:
            # Cold-start: no prior attempt to reference.
            return ""

        outcome = parent.outcome
        if isinstance(outcome, InfrastructureFailureFeedback):
            # Treat as cold-start — the infra crash was not the model's
            # fault and there is no actionable signal to surface.
            return ""

        prompt = "Here is your latest implementation:\n"
        prompt += f"```python\n{parent.code}\n```\n\n"
        prompt += "Your custom_kernel was evaluated on GPU.\n\nHere is the evaluation result:\n"

        match outcome.kind:
            case "compile_failed":
                assert isinstance(outcome, CompileFailedFeedback)
                prompt += "Your kernel failed to compile.\n\n"
                prompt += "Compilation error:\n"
                prompt += (
                    f"{_truncate_head(outcome.compilation_error, self._max_compilation_error_chars)}\n\n"
                )
                prompt += "Please fix the errors and try again."
            case "runtime_error":
                assert isinstance(outcome, RuntimeErrorFeedback)
                prompt += "Your kernel raised an exception at runtime.\n\n"
                prompt += f"Error type: {outcome.runtime_error_name}\n\n"
                prompt += "Error message:\n"
                prompt += (
                    f"{_truncate_head(outcome.runtime_error, self._max_runtime_error_chars)}\n\n"
                )
                prompt += "Traceback:\n"
                prompt += f"{_truncate_tail(outcome.traceback, self._max_traceback_chars)}\n\n"
                prompt += "Please fix the errors and try again."
            case "incorrect":
                assert isinstance(outcome, IncorrectFeedback)
                prompt += "Your kernel produced incorrect output compared to the reference.\n\n"
                prompt += "Correctness issue:\n"
                prompt += (
                    f"{_truncate_head(outcome.error_message, self._max_incorrect_error_chars)}\n\n"
                )
                prompt += "Please fix the correctness issues and try again."
            case "success":
                assert isinstance(outcome, SuccessFeedback)
                prompt += "You are iteratively optimizing runtime (microseconds).\n\n"
                prompt += (
                    f"Your kernel is correct. "
                    f"Aggregated speedup: {outcome.aggregated_speedup:.3f}x "
                    f"(aggregation method: {outcome.aggregation_method}).\n\n"
                )
                sorted_cases = sorted(
                    outcome.per_case_speedups, key=lambda c: c.speedup
                )
                prompt += "Per-case breakdown (slowest first):\n"
                for case in sorted_cases:
                    ref_us = case.ref_runtime_ns / 1_000.0
                    candidate_us = case.runtime_ns / 1_000.0
                    prompt += (
                        f"  seqlen={case.seqlen}, bs={case.bs}, dim={case.dim}, "
                        f"hiddendim={case.hiddendim}, nomask={case.nomask}, "
                        f"dist={case.distribution}: "
                        f"{case.speedup:.3f}x "
                        f"(ref: {ref_us:.1f}μs, candidate: {candidate_us:.1f}μs)\n"
                    )
                prompt += (
                    "\nPlease rewrite the entire kernel to be as fast as possible. "
                    "Focus on the slowest configurations listed above."
                )
            case "parse_failure":
                assert isinstance(outcome, ParseFailureFeedback)
                prompt += "Your previous response did not contain an extractable python code block.\n\n"
                prompt += f"Reason: {outcome.reason}\n\n"
                prompt += (
                    "Remember: put your complete kernel in a single trailing "
                    "```python ... ``` block."
                )

        return prompt


_ = implements(FeedbackPromptRenderer)(TriMulFeedbackPromptRenderer)
