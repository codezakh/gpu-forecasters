"""Invariant tests for the v2 RL env.

Drives a real ``GptOssRenderer`` end-to-end with a hand-built
assistant message that contains a valid tool call, plus a parse-failure
case where the model emitted only natural-language text. This catches
prompt-build / parse-response wiring breakage; it does *not* train
anything.
"""

from __future__ import annotations

import asyncio

import pytest
from tinker_cookbook.renderers import Message, ToolCall, get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

from arid_badger.landscape_map.v2 import (
    HardwareContext,
    KernelBinPredictionEnv,
    KernelRuntimeEstimate,
    LabeledKernelItem,
    RewardComponents,
    RewardFunction,
    SpeedupBin,
)


# Skip the whole module if the gpt-oss tokenizer / renderer can't be
# resolved (e.g. no network on a machine without HF cache). These
# tests are smoke-coverage for the renderer wiring; if you need them
# to run, prime the HF cache once.
pytest.importorskip("transformers")


def _hardware() -> HardwareContext:
    return HardwareContext(
        device_name="A100-SXM4-80GB",
        compute_capability=(8, 0),
        total_global_memory_gb=79.2,
        multiprocessor_count=108,
        max_threads_per_multiprocessor=2048,
        clock_rate_ghz=1.41,
        memory_clock_rate_ghz=1.59,
        memory_bus_width_bits=5120,
    )


def _item() -> LabeledKernelItem:
    return LabeledKernelItem(
        pack_name="test_pack",
        anchor_source="seed",
        anchor_code="def f(): pass",
        candidate_code="def f(): pass",
        speedup_geomean=1.5,  # → MINOR_SPEEDUP (bin 5) under from_speedup
        hardware=_hardware(),
        source_id="test-id",
    )


class _FixedReward(RewardFunction):
    """Reward that exposes whatever the env handed it."""

    def __init__(self) -> None:
        self.last_estimate: KernelRuntimeEstimate | None = None
        self.last_truth: SpeedupBin | None = None

    def reward(
        self,
        estimate: KernelRuntimeEstimate | None,
        true_bin: SpeedupBin,
    ) -> RewardComponents:
        self.last_estimate = estimate
        self.last_truth = true_bin
        if estimate is None:
            return RewardComponents(distance=0.0, calibration=0.0, total=0.0)
        return RewardComponents(distance=0.5, calibration=0.5, total=0.5)


def _make_env(item: LabeledKernelItem, reward: RewardFunction) -> KernelBinPredictionEnv:
    tokenizer = get_tokenizer("openai/gpt-oss-20b")
    renderer = get_renderer("gpt_oss_medium_reasoning", tokenizer=tokenizer)
    return KernelBinPredictionEnv(item=item, renderer=renderer, reward_fn=reward)


def _valid_tool_call_message() -> Message:
    """Hand-build the message the renderer would emit for a valid tool call."""
    args = (
        '{"predicted_bin": 5, '
        '"p_severe_slowdown": 0.01, "p_significant_slowdown": 0.04, '
        '"p_moderate_slowdown": 0.05, "p_minor_slowdown": 0.10, '
        '"p_minor_speedup": 0.50, "p_significant_speedup": 0.20, '
        '"p_high_speedup": 0.05, "p_extreme_speedup": 0.05, '
        '"reasoning": "test"}'
    )
    return Message(
        role="assistant",
        content="",
        tool_calls=[
            ToolCall(
                function=ToolCall.FunctionBody(
                    name="submit_kernel_runtime_estimate", arguments=args
                )
            )
        ],
    )


def test_initial_observation_includes_tool_prefix_and_user_prompt():
    """The prompt must contain both the tool definition and the user query."""
    item = _item()
    env = _make_env(item, _FixedReward())
    prompt, stops = asyncio.run(env.initial_observation())
    # Two stop tokens for gpt-oss: <|return|> and <|call|>.
    assert len(stops) == 2
    # The tokenized prompt should be non-trivial — Harmony tool prefix
    # alone is hundreds of tokens.
    assert prompt.length > 100


def test_step_extracts_estimate_and_logs_reward():
    """Happy path: the env hands the parsed estimate to the reward fn."""
    item = _item()
    reward = _FixedReward()
    env = _make_env(item, reward)
    # Run initial_observation to mirror real lifecycle, even though we
    # don't use the returned tokens.
    asyncio.run(env.initial_observation())

    # We feed the env a hand-built parsed message rather than tokens
    # from a real sample. Trick: monkeypatch parse_response so the env
    # sees our message and (parse_success=True).
    msg = _valid_tool_call_message()
    env._renderer.parse_response = lambda response: (msg, True)  # type: ignore[method-assign]

    result = asyncio.run(env.step(action=[1, 2, 3]))
    assert reward.last_truth == SpeedupBin.MINOR_SPEEDUP
    assert reward.last_estimate is not None
    assert reward.last_estimate.predicted_bin == SpeedupBin.MINOR_SPEEDUP
    assert result.reward == 0.5
    assert result.episode_done is True
    # Logged metrics include the components and the parsing flag.
    assert result.metrics["parsed"] == 1.0
    assert result.metrics["exact_match"] == 1.0


def test_step_handles_parse_failure_with_none_estimate():
    """When the model emits no tool call, reward fn gets ``None``."""
    item = _item()
    reward = _FixedReward()
    env = _make_env(item, reward)
    asyncio.run(env.initial_observation())

    text_only = Message(role="assistant", content="I don't know.")
    env._renderer.parse_response = lambda response: (text_only, True)  # type: ignore[method-assign]

    result = asyncio.run(env.step(action=[1, 2, 3]))
    assert reward.last_estimate is None
    assert result.reward == 0.0
    assert result.metrics["parsed"] == 0.0
