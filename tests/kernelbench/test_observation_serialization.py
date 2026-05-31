"""Verify that Evaluation[KernelBenchObservation] round-trips through JSON
for every variant of KernelBenchFeedback.

This guards against adding a new feedback variant that silently breaks
checkpoint serialization at runtime.
"""

import json

import pytest

from gpu_forecasters.hill_climbing.domain import Evaluation
from gpu_forecasters.hill_climbing.scoring_providers.kernelbench import (
    KernelBenchObservation,
)
from gpu_forecasters.kernelbench.core import (
    CompileFailedFeedback,
    IncorrectFeedback,
    InfrastructureFailureFeedback,
    RuntimeErrorFeedback,
    SuccessFeedback,
)


FEEDBACK_CASES = [
    pytest.param(
        CompileFailedFeedback(
            compilation_error_name="ModuleNotFoundError",
            compilation_error="No module named 'triton'",
        ),
        id="compile_failed",
    ),
    pytest.param(
        RuntimeErrorFeedback(
            runtime_error_name="RuntimeError",
            runtime_error="CUDA error: illegal memory access",
            runtime_error_traceback="Traceback ...",
        ),
        id="runtime_error",
    ),
    pytest.param(
        IncorrectFeedback(
            correctness_issue="Output mismatch",
            max_difference=["0.1", "0.2"],
            avg_difference=["0.05"],
        ),
        id="incorrect",
    ),
    pytest.param(
        SuccessFeedback(
            runtime_us=100.0,
            ref_runtime_us=200.0,
            speedup=2.0,
        ),
        id="success",
    ),
    pytest.param(
        InfrastructureFailureFeedback(
            reason="Timed out after 300s",
        ),
        id="infrastructure_failure",
    ),
]


@pytest.mark.parametrize("feedback", FEEDBACK_CASES)
def test_evaluation_round_trips_through_json(feedback):
    observation = KernelBenchObservation(feedback=feedback)
    evaluation = Evaluation[KernelBenchObservation](observation=observation, reward=1.5)

    json_str = evaluation.model_dump_json()
    # Ensure it's valid JSON.
    json.loads(json_str)

    restored = Evaluation[KernelBenchObservation].model_validate_json(json_str)
    assert restored == evaluation


@pytest.mark.parametrize("feedback", FEEDBACK_CASES)
def test_evaluation_with_none_reward_round_trips(feedback):
    observation = KernelBenchObservation(feedback=feedback)
    evaluation = Evaluation[KernelBenchObservation](observation=observation, reward=None)

    json_str = evaluation.model_dump_json()
    restored = Evaluation[KernelBenchObservation].model_validate_json(json_str)
    assert restored == evaluation
    assert restored.reward is None
