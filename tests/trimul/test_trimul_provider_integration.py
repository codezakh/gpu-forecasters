"""Provider-level Modal smoke test.

Verifies ``TriMulModalProvider`` plugs into the ``EvaluationProvider``
protocol end-to-end: lifecycle, thread-pool batch_evaluate, reward
shaping, invocation sink records.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from arid_badger.hill_climbing.scoring_providers.trimul_modal import (
    TriMulModalProvider,
)
from arid_badger.trimul.cases import BENCHMARK_CASES


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

    with TriMulModalProvider(test_args=BENCHMARK_CASES[0]) as provider:
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
