from types import SimpleNamespace

import pytest
from ulid import ULID

from arid_badger.greedy_search.domain import MutationContext
from arid_badger.greedy_search.domain import (
    CompileFailedFeedback,
    SuccessFeedback,
    ValidEvaluation,
)
from arid_badger.greedy_search.feedback_mutation import (
    KernelBenchExecutionFeedbackMutationFunction,
    format_feedback_mutation_prompt,
)


def test_format_feedback_mutation_prompt_raises_without_feedback() -> None:
    feedback = CompileFailedFeedback(
        compilation_error_name="CompilerError",
        compilation_error="Failed to compile",
    )
    with pytest.raises(ValueError):
        format_feedback_mutation_prompt(
            base_prompt="BASE",
            previous_kernel_code="",
            feedback=feedback,
        )

    with pytest.raises(ValueError):
        format_feedback_mutation_prompt(
            base_prompt="",
            previous_kernel_code="code",
            feedback=feedback,
        )

    with pytest.raises(ValueError):
        format_feedback_mutation_prompt(
            base_prompt="BASE",
            previous_kernel_code="code",
            feedback=None,  # type: ignore[arg-type]
        )


def test_format_feedback_mutation_prompt_compile_failed_includes_error_and_instruction() -> (
    None
):
    feedback = CompileFailedFeedback(
        compilation_error_name="CompilerError",
        compilation_error="Failed to compile",
    )
    prompt = format_feedback_mutation_prompt(
        base_prompt="BASE_PROMPT",
        previous_kernel_code="def kernel(): pass",
        feedback=feedback,
    )
    assert "BASE_PROMPT" in prompt
    assert "Here is your latest generation" in prompt
    assert "CompilerError" in prompt
    assert "Failed to compile" in prompt
    assert "Please fix the errors" in prompt


def test_feedback_mutation_function_sends_formatted_prompt_and_parses_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured = {}

    def fake_completion(**kwargs):
        captured["prompt"] = kwargs["messages"][0]["content"]
        content = "```python\nclass ModelNew:\n    pass\n```"
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )

    monkeypatch.setattr(
        "arid_badger.greedy_search.feedback_mutation.get_prompt_for_backend",
        lambda **_: "BASE_PROMPT",
    )
    monkeypatch.setattr(
        "arid_badger.greedy_search.feedback_mutation.completion",
        fake_completion,
    )

    feedback = SuccessFeedback(runtime_us=1.0, ref_runtime_us=2.0, speedup=2.0)
    evaluation = ValidEvaluation(speedup=2.0, metrics=None, execution_feedback=feedback)
    previous_ulid = ULID()
    context = MutationContext(
        reference_kernel_code="ref",
        previous_kernel_code="previous",
        previous_kernel_ulid=previous_ulid,
        previous_evaluation=evaluation,
    )

    mutated = KernelBenchExecutionFeedbackMutationFunction()(context)
    assert "BASE_PROMPT" in captured["prompt"]
    assert "previous" in captured["prompt"]
    assert "Speedup" in captured["prompt"]
    assert mutated.kernel_code
    assert mutated.ulid
    assert mutated.ancestor_ulid == previous_ulid
