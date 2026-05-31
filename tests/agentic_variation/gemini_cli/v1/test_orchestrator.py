"""Unit tests for orchestrator pure helpers (settings builder, cache key)."""

from __future__ import annotations

from gpu_forecasters.agentic_variation.gemini_cli.v1.models import (
    ExperimentConfig,
    ThinkingLevel,
)
from gpu_forecasters.agentic_variation.gemini_cli.v1.orchestrator import (
    _baseline_cache_key,
    _build_gemini_settings,
)


def _config(
    *,
    model_slug: str = "gemini-3-flash-preview",
    max_session_turns: int = 10,
    thinking_level: ThinkingLevel | None = None,
) -> ExperimentConfig:
    return ExperimentConfig(
        model_slug=model_slug,
        gpu="H100",
        triton_version="3.3.1",
        max_session_turns=max_session_turns,
        aggregator="geomean",
        thinking_level=thinking_level,
    )


def test_build_gemini_settings_base_shape() -> None:
    settings = _build_gemini_settings(
        _config(max_session_turns=17), "http://127.0.0.1:9999/mcp/"
    )
    assert settings["model"] == {"maxSessionTurns": 17}
    mcp_servers = settings["mcpServers"]
    assert mcp_servers["trimul"]["httpUrl"] == "http://127.0.0.1:9999/mcp/"
    assert mcp_servers["trimul"]["trust"] is True
    # No thinking override when thinking_level is None.
    assert "modelConfigs" not in settings


def test_build_gemini_settings_thinking_level_pro_emits_both_aliases() -> None:
    settings = _build_gemini_settings(
        _config(model_slug="gemini-3-pro-preview", thinking_level="LOW"),
        "http://127.0.0.1:1234/mcp/",
    )
    overrides = settings["modelConfigs"]["overrides"]
    matched_models = [entry["match"]["model"] for entry in overrides]
    # Both the CLI-arg id and the resolver-internal id need override
    # entries so the thinkingLevel takes effect regardless of which id
    # the CLI resolves to at request time.
    assert matched_models == ["gemini-3-pro-preview", "gemini-3.1-pro-preview"]
    for entry in overrides:
        level = entry["modelConfig"]["generateContentConfig"]["thinkingConfig"][
            "thinkingLevel"
        ]
        assert level == "LOW"


def test_build_gemini_settings_thinking_level_unknown_model_single_entry() -> None:
    # Models not in the alias map get a single override matched on their
    # own slug (no silent expansion).
    settings = _build_gemini_settings(
        _config(model_slug="some-future-model", thinking_level="HIGH"),
        "http://127.0.0.1:1234/mcp/",
    )
    overrides = settings["modelConfigs"]["overrides"]
    assert len(overrides) == 1
    assert overrides[0]["match"]["model"] == "some-future-model"


def test_baseline_cache_key_incorporates_each_field() -> None:
    a = _config()
    b = ExperimentConfig(
        model_slug=a.model_slug,
        gpu="A100",  # only gpu differs
        triton_version=a.triton_version,
        max_session_turns=a.max_session_turns,
        aggregator=a.aggregator,
        thinking_level=a.thinking_level,
    )
    # Different GPU → different cache key.
    assert _baseline_cache_key(a) != _baseline_cache_key(b)
    # Model slug and thinking level do NOT go into the key — baseline is
    # the seed's verdict under a given scoring config, not a given agent.
    c = ExperimentConfig(
        model_slug="different-model",
        gpu=a.gpu,
        triton_version=a.triton_version,
        max_session_turns=a.max_session_turns,
        aggregator=a.aggregator,
        thinking_level="HIGH",
    )
    assert _baseline_cache_key(a) == _baseline_cache_key(c)
    # Key must be a single filename (no path separators) so FileCache
    # doesn't treat it as a nested subdirectory.
    assert "/" not in _baseline_cache_key(a)
