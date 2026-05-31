"""Provider-level Modal smoke test.

Verifies ``TriMulModalProvider`` plugs into the ``EvaluationProvider``
protocol end-to-end: lifecycle, thread-pool batch_evaluate, reward
shaping, invocation sink records.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from gpu_forecasters.hill_climbing.scoring_providers.trimul_modal import (
    TriMulModalProvider,
)
from gpu_forecasters.trimul.cases import BENCHMARK_CASES
from gpu_forecasters.trimul.core import SuccessFeedback


_FIXTURES = Path(__file__).parent / "fixtures"


pytestmark = [pytest.mark.modal, pytest.mark.integration]


_ZEROS_CANDIDATE = """\
import torch

def custom_kernel(data):
    input_tensor = data[0]
    return torch.zeros_like(input_tensor)
"""


def test_provider_batch_evaluate_end_to_end() -> None:
    starter = (_FIXTURES / "reference_submission.py").read_text()
    leaderboard = (_FIXTURES / "leaderboard" / "ttt-discover.py").read_text()
    candidates = [starter, _ZEROS_CANDIDATE, leaderboard]

    # Two nomask=True benchmark cases — enough to exercise multi-case
    # aggregation without spending too much GPU time.
    test_cases = [BENCHMARK_CASES[0], BENCHMARK_CASES[3]]

    with TriMulModalProvider(test_cases=test_cases) as provider:
        results = provider.batch_evaluate(candidates)

    assert len(results) == 3
    starter_eval, zeros_eval, leaderboard_eval = results

    assert starter_eval.reward is not None
    assert zeros_eval.reward is None
    assert leaderboard_eval.reward is not None
    assert leaderboard_eval.reward > starter_eval.reward, (
        f"leaderboard reward {leaderboard_eval.reward} should exceed "
        f"starter {starter_eval.reward}"
    )

    # Verify the success observation carries per-case breakdown.
    assert isinstance(leaderboard_eval.observation.feedback, SuccessFeedback)
    assert len(leaderboard_eval.observation.feedback.per_case_speedups) == 2
    assert len(leaderboard_eval.observation.per_case_results) == 2
