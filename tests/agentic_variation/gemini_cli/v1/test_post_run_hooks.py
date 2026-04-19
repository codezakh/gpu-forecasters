"""Unit tests for the post-run hook dispatch mechanism.

Exercises ``dispatch_post_run_hooks`` without Docker / Modal / Gemini —
the dispatch is a pure loop over ``config.post_run_hooks``, so these
tests prove the config wiring actually invokes custom hooks with the
right ``PostRunContext``.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from arid_badger.agentic_variation.gemini_cli.v1 import (
    ExperimentConfig,
    PostRunContext,
    PostRunHook,
    dispatch_post_run_hooks,
)


def _base_config() -> ExperimentConfig:
    return ExperimentConfig(
        model_slug="gemini-3-flash-preview",
        gpu="H100",
        triton_version="3.3.1",
        max_session_turns=10,
        aggregator="geomean",
    )


def _with_hooks(hooks: tuple[PostRunHook, ...]) -> ExperimentConfig:
    return dataclasses.replace(_base_config(), post_run_hooks=hooks)


def test_default_hook_copies_kernel_files_only(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    artifacts = tmp_path / "artifacts"
    scratch.mkdir()
    artifacts.mkdir()
    _ = (scratch / "kernel.py").write_text("# final")
    _ = (scratch / "kernel_v1.py").write_text("# v1")
    _ = (scratch / "notes.md").write_text("ignored")

    dispatch_post_run_hooks(_base_config(), scratch, artifacts)

    assert (artifacts / "kernel.py").read_text() == "# final"
    assert (artifacts / "kernel_v1.py").read_text() == "# v1"
    assert not (artifacts / "notes.md").exists()


def test_custom_hook_receives_expected_context(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    artifacts = tmp_path / "artifacts"
    scratch.mkdir()
    artifacts.mkdir()

    captured: list[PostRunContext] = []

    def marker_hook(ctx: PostRunContext) -> None:
        captured.append(ctx)
        _ = (ctx.run_artifacts_dir / "marker").write_text("ran")

    config = _with_hooks((marker_hook,))
    dispatch_post_run_hooks(config, scratch, artifacts)

    assert (artifacts / "marker").read_text() == "ran"
    assert len(captured) == 1
    assert captured[0].scratch == scratch
    assert captured[0].run_artifacts_dir == artifacts
    assert captured[0].config is config


def test_multiple_hooks_run_in_declared_order(tmp_path: Path) -> None:
    scratch = tmp_path / "scratch"
    artifacts = tmp_path / "artifacts"
    scratch.mkdir()
    artifacts.mkdir()

    order: list[str] = []

    def first(ctx: PostRunContext) -> None:
        del ctx
        order.append("first")

    def second(ctx: PostRunContext) -> None:
        del ctx
        order.append("second")

    dispatch_post_run_hooks(_with_hooks((first, second)), scratch, artifacts)

    assert order == ["first", "second"]
