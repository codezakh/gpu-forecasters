"""Integration test for the full Gemini-CLI-on-TriMul stack.

Runs ``run_experiment`` with a 3-turn budget against real Docker + Modal
+ Gemini. Not collected by default; gated by ``integration`` and
``modal`` markers. Run with::

    cd 15-arid-badger
    uv run --env-file .env pytest -m "integration and modal" \\
        tests/agentic_variation/gemini_cli/v1/test_run_experiment_integration.py -v
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from gpu_forecasters.agentic_variation.gemini_cli.v1 import (
    ExperimentConfig,
    RESULT_FILENAME,
    TRAJECTORY_FILENAME,
    TrimulRunResult,
    load_trajectory,
    run_experiment,
)

pytestmark = [pytest.mark.integration, pytest.mark.modal]


def test_run_experiment_end_to_end(tmp_path: Path) -> None:
    if not os.environ.get("GEMINI_API_KEY"):
        pytest.skip("GEMINI_API_KEY not set")

    config = ExperimentConfig(
        model_slug="gemini-3-flash-preview",
        gpu="H100",
        triton_version="3.3.1",
        max_session_turns=3,
        aggregator="geomean",
        thinking_level=None,
    )
    run_dir = tmp_path / "run_integration"
    run_dir.mkdir()

    result = run_experiment(config, run_dir)

    assert isinstance(result, TrimulRunResult)
    assert result.elapsed_s > 0

    result_path = run_dir / RESULT_FILENAME
    assert result_path.is_file()
    reloaded = TrimulRunResult.model_validate_json(result_path.read_text())
    assert reloaded == result

    traj_path = run_dir / TRAJECTORY_FILENAME
    assert traj_path.is_file()
    records = load_trajectory(run_dir)
    assert len(records) >= 1, "agent did not call score_trimul even once"

    for fname in (
        "system_prompt.md",
        "user_prompt.md",
        "seed_kernel.py",
        "baseline_feedback.json",
        "server.log",
        "agent_raw.log",
    ):
        assert (run_dir / fname).is_file(), f"missing {fname}"

    baseline_cache = run_dir.parent / "baseline_cache"
    assert baseline_cache.is_dir()
    assert any(baseline_cache.iterdir()), "baseline cache empty"

    assert not (run_dir / ".server.pid").is_file()
