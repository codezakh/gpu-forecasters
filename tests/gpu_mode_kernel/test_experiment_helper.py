"""Unit tests for the generic experiment helper.

Heavy I/O paths (Modal, litellm) are exercised by the per-pack smoke
runs in `experiments/e007{7,8,9}/`. These tests cover the
helper's pure logic: result loader on missing/present directories,
summary serialization round-trip, and the typed schemas.
"""

from __future__ import annotations

from pathlib import Path

from arid_badger.gpu_mode_kernel.experiment_helper import (
    ExperimentConfig,
    ProviderConfig,
    RunConfig,
    RunSummary,
    load_run_summaries,
)
from arid_badger.max_reward_puct.v2.config import SearchConfig as V2SearchConfig


def _example_config() -> ExperimentConfig:
    return ExperimentConfig(
        num_runs=3,
        run=RunConfig(
            search=V2SearchConfig(
                total_budget_steps=40,
                batch_size=2,
                samples_per_parent=4,
                k_per_parent=2,
                archive_capacity=1000,
                c_puct=1.0,
                per_request_timeout_s=900.0,
            ),
            provider=ProviderConfig(
                model_slug="gemini/gemini-3-flash-preview",
                gpu="A100-80GB",
                aggregator="geomean",
                max_llm_concurrency=8,
                max_tokens=None,
                request_timeout_s=900.0,
            ),
        ),
    )


def test_load_run_summaries_returns_empty_for_missing_dir(tmp_path: Path) -> None:
    """Loader must not raise on a fresh experiment that has produced
    nothing yet — that's the common case when comparison code spans
    two experiments and one has not yet been run."""
    assert load_run_summaries(tmp_path / "nonexistent") == []


def test_load_run_summaries_reads_runs_in_lexical_order(tmp_path: Path) -> None:
    """``run_NN/`` ordering matters when comparing curves across runs;
    sort by name (which equals sort-by-index given the zero-padding)."""
    summaries = [
        RunSummary(
            steps_completed=10 + i,
            archive_size=5 + i,
            best_reward=2.0 + i * 0.5,
            seed_reward=1.0,
            best_node_ulid=f"01ABC{i}",
            num_evaluations_total=20,
            num_evaluations_correct=18,
            num_evaluation_failures=1,
            num_mutation_failures=1,
            wall_clock_seconds=300.0 + i,
        )
        for i in range(3)
    ]
    for i, s in enumerate(summaries):
        run_dir = tmp_path / f"run_{i:02d}"
        run_dir.mkdir()
        _ = (run_dir / "summary.json").write_text(s.model_dump_json())

    loaded = load_run_summaries(tmp_path)

    assert [s.best_reward for s in loaded] == [2.0, 2.5, 3.0]
    assert [s.steps_completed for s in loaded] == [10, 11, 12]


def test_experiment_config_uses_the_published_v2_search_config_type() -> None:
    """The helper's RunConfig embeds ``arid_badger.max_reward_puct.v2.config.SearchConfig``
    — the same type the v2 driver consumes — so a config the helper
    accepts can be threaded straight to a SearchDriver without
    rebuilding."""
    cfg = _example_config()
    assert isinstance(cfg.run.search, V2SearchConfig)
    assert cfg.run.search.total_budget_steps == 40
