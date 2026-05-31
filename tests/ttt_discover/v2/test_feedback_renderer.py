import pytest

from gpu_forecasters.trimul.core import (
    CaseSpeedup,
    CompileFailedFeedback,
    IncorrectFeedback,
    InfrastructureFailureFeedback,
    RuntimeErrorFeedback,
    SuccessFeedback,
)
from gpu_forecasters.ttt_discover.v2.domain.candidate import Candidate, CandidateId
from gpu_forecasters.ttt_discover.v2.domain.context import FeedbackPromptContext
from gpu_forecasters.ttt_discover.v2.domain.outcome import (
    ParseFailureFeedback,
    TriMulRLOutcome,
)
from gpu_forecasters.ttt_discover.v2.domain.problem import TriMulProblem
from gpu_forecasters.ttt_discover.v2.renderers.feedback_trimul import (
    TriMulFeedbackPromptRenderer,
)


def _problem() -> TriMulProblem:
    return TriMulProblem(
        base_prompt_text="x",
        test_cases=(),
        gpu_name="A100-80GB",
        triton_version="3.3.1",
        target_runtime_us=2500.0,
    )


def _parent_with(outcome: TriMulRLOutcome) -> Candidate:
    return Candidate(
        id=CandidateId("p1"),
        code="def custom_kernel(data): return data[0]",
        timestep=0,
        parent_id=None,
        outcome=outcome,
        reward=0.0,
    )


def test_cold_start_is_empty() -> None:
    ctx = FeedbackPromptContext(problem=_problem(), parent=None)
    assert TriMulFeedbackPromptRenderer().render(ctx) == ""


def test_parent_with_infra_failure_is_empty() -> None:
    parent = _parent_with(InfrastructureFailureFeedback(reason="modal"))
    ctx = FeedbackPromptContext(problem=_problem(), parent=parent)
    assert TriMulFeedbackPromptRenderer().render(ctx) == ""


def test_compile_failure_mentions_compile_error() -> None:
    parent = _parent_with(
        CompileFailedFeedback(compilation_error="SyntaxError: unexpected token")
    )
    ctx = FeedbackPromptContext(problem=_problem(), parent=parent)
    rendered = TriMulFeedbackPromptRenderer().render(ctx)
    assert "failed to compile" in rendered
    assert "SyntaxError" in rendered
    assert "def custom_kernel" in rendered


def test_runtime_error_mentions_traceback() -> None:
    parent = _parent_with(
        RuntimeErrorFeedback(
            runtime_error_name="ValueError",
            runtime_error="bad shape",
            traceback="Traceback:\n  bar.py:1",
        )
    )
    rendered = TriMulFeedbackPromptRenderer().render(
        FeedbackPromptContext(problem=_problem(), parent=parent)
    )
    assert "ValueError" in rendered
    assert "Traceback" in rendered


def test_incorrect_mentions_error_message() -> None:
    parent = _parent_with(IncorrectFeedback(error_message="max abs err 1.2e-1"))
    rendered = TriMulFeedbackPromptRenderer().render(
        FeedbackPromptContext(problem=_problem(), parent=parent)
    )
    assert "incorrect" in rendered
    assert "max abs err" in rendered


def test_success_lists_per_case_speedups() -> None:
    parent = _parent_with(
        SuccessFeedback(
            aggregated_speedup=1.5,
            aggregation_method="geomean",
            per_case_speedups=[
                CaseSpeedup(
                    seqlen=256,
                    bs=2,
                    dim=128,
                    hiddendim=128,
                    nomask=True,
                    distribution="normal",
                    speedup=2.0,
                    runtime_ns=1_000_000.0,
                    ref_runtime_ns=2_000_000.0,
                ),
                CaseSpeedup(
                    seqlen=1024,
                    bs=1,
                    dim=128,
                    hiddendim=128,
                    nomask=True,
                    distribution="normal",
                    speedup=0.5,
                    runtime_ns=4_000_000.0,
                    ref_runtime_ns=2_000_000.0,
                ),
            ],
        )
    )
    rendered = TriMulFeedbackPromptRenderer().render(
        FeedbackPromptContext(problem=_problem(), parent=parent)
    )
    # Slowest-first ordering.
    assert rendered.index("seqlen=1024") < rendered.index("seqlen=256")
    assert "2.000x" in rendered
    assert "geomean" in rendered


def test_parse_failure_suggests_codeblock() -> None:
    parent = _parent_with(ParseFailureFeedback(reason="nothing found"))
    rendered = TriMulFeedbackPromptRenderer().render(
        FeedbackPromptContext(problem=_problem(), parent=parent)
    )
    assert "python" in rendered.lower()
    assert "nothing found" in rendered


def test_truncation_budgets_are_applied() -> None:
    renderer = TriMulFeedbackPromptRenderer(max_compilation_error_chars=10)
    parent = _parent_with(CompileFailedFeedback(compilation_error="A" * 100))
    rendered = renderer.render(FeedbackPromptContext(problem=_problem(), parent=parent))
    assert "[truncated]" in rendered


@pytest.mark.parametrize(
    "outcome",
    [
        CompileFailedFeedback(compilation_error="x"),
        RuntimeErrorFeedback(runtime_error_name="E", runtime_error="x", traceback="y"),
        IncorrectFeedback(error_message="x"),
    ],
)
def test_feedback_references_parent_code(outcome: TriMulRLOutcome) -> None:
    parent = _parent_with(outcome)
    rendered = TriMulFeedbackPromptRenderer().render(
        FeedbackPromptContext(problem=_problem(), parent=parent)
    )
    assert "def custom_kernel" in rendered
