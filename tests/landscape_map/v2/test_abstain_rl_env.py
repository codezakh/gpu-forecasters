"""Invariant tests for the v2 abstain RL env.

Drives a real ``GptOssRenderer`` end-to-end with hand-built assistant
messages — one for each of the predict / defer / parse-failure /
multi-tool-call / unknown-tool arms. Verifies that the env hands the
right :class:`PredictOrDefer` value (or ``None``) to the reward
function and reports the right metrics.
"""

from __future__ import annotations

import asyncio

import pytest
from tinker_cookbook.renderers import Message, ToolCall, get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer

from arid_badger.landscape_map.v2 import (
    AbstainRewardComponents,
    AbstainRewardFunction,
    Deferral,
    Forecast,
    HardwareContext,
    KernelBinPredictionAbstainEnv,
    LabeledKernelItem,
    PredictOrDefer,
    SpeedupBin,
)


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
        speedup_geomean=1.5,  # → MINOR_SPEEDUP (bin 5)
        hardware=_hardware(),
        source_id="test-id",
    )


class _FixedReward(AbstainRewardFunction):
    """Reward that exposes whatever the env handed it."""

    def __init__(self) -> None:
        self.last_outcome: PredictOrDefer | None = None
        self.last_truth: SpeedupBin | None = None

    def reward(
        self,
        outcome: PredictOrDefer | None,
        true_bin: SpeedupBin,
    ) -> AbstainRewardComponents:
        self.last_outcome = outcome
        self.last_truth = true_bin
        if outcome is None:
            return AbstainRewardComponents(
                correctness=0.0, abstention=0.0, total=0.0
            )
        if outcome.kind == "defer":
            return AbstainRewardComponents(
                correctness=0.0, abstention=0.2, total=0.2
            )
        # Forecast arm: pretend right.
        return AbstainRewardComponents(
            correctness=1.0, abstention=1.0, total=2.0
        )


def _make_env(
    item: LabeledKernelItem, reward: AbstainRewardFunction
) -> KernelBinPredictionAbstainEnv:
    tokenizer = get_tokenizer("openai/gpt-oss-20b")
    renderer = get_renderer("gpt_oss_medium_reasoning", tokenizer=tokenizer)
    return KernelBinPredictionAbstainEnv(
        item=item, renderer=renderer, reward_fn=reward
    )


def _predict_message() -> Message:
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


def _defer_message(reason: str = "novel hardware shape") -> Message:
    args = '{"reason": "' + reason + '"}'
    return Message(
        role="assistant",
        content="",
        tool_calls=[
            ToolCall(
                function=ToolCall.FunctionBody(
                    name="defer_to_real_evaluator", arguments=args
                )
            )
        ],
    )


def test_initial_observation_includes_both_tools_and_user_prompt() -> None:
    env = _make_env(_item(), _FixedReward())
    prompt, stops = asyncio.run(env.initial_observation())
    assert len(stops) == 2
    # Two tools registered → tool-prefix is even longer than the predict-only env.
    assert prompt.length > 200


def test_step_extracts_forecast_outcome_and_logs_correctness() -> None:
    item = _item()
    reward = _FixedReward()
    env = _make_env(item, reward)
    asyncio.run(env.initial_observation())

    msg = _predict_message()
    env._renderer.parse_response = lambda response: (msg, True)  # type: ignore[method-assign]

    result = asyncio.run(env.step(action=[1, 2, 3]))
    assert reward.last_truth == SpeedupBin.MINOR_SPEEDUP
    assert isinstance(reward.last_outcome, Forecast)
    assert reward.last_outcome.estimate.predicted_bin == SpeedupBin.MINOR_SPEEDUP
    assert result.reward == 2.0
    assert result.episode_done is True
    assert result.metrics["parsed"] == 1.0
    assert result.metrics["forecasted"] == 1.0
    assert result.metrics["deferred"] == 0.0
    assert result.metrics["exact_match"] == 1.0


def test_step_extracts_deferral_outcome_and_logs_abstention() -> None:
    item = _item()
    reward = _FixedReward()
    env = _make_env(item, reward)
    asyncio.run(env.initial_observation())

    msg = _defer_message(reason="too uncertain about target hardware")
    env._renderer.parse_response = lambda response: (msg, True)  # type: ignore[method-assign]

    result = asyncio.run(env.step(action=[1, 2, 3]))
    assert isinstance(reward.last_outcome, Deferral)
    assert reward.last_outcome.reason == "too uncertain about target hardware"
    assert result.reward == 0.2
    assert result.metrics["parsed"] == 1.0
    assert result.metrics["forecasted"] == 0.0
    assert result.metrics["deferred"] == 1.0
    # Deferred rollouts have no predicted bin.
    assert result.metrics["exact_match"] == 0.0


def test_step_handles_no_tool_call_as_parse_failure() -> None:
    item = _item()
    reward = _FixedReward()
    env = _make_env(item, reward)
    asyncio.run(env.initial_observation())

    text_only = Message(role="assistant", content="I don't know.")
    env._renderer.parse_response = lambda response: (text_only, True)  # type: ignore[method-assign]

    result = asyncio.run(env.step(action=[1, 2, 3]))
    assert reward.last_outcome is None
    assert result.reward == 0.0
    assert result.metrics["parsed"] == 0.0
    assert result.metrics["forecasted"] == 0.0
    assert result.metrics["deferred"] == 0.0


def test_step_treats_multiple_tool_calls_as_parse_failure() -> None:
    """The contract is exactly one tool call. Two = ambiguous → parse failure."""
    item = _item()
    reward = _FixedReward()
    env = _make_env(item, reward)
    asyncio.run(env.initial_observation())

    predict_args = (
        '{"predicted_bin": 5, '
        '"p_severe_slowdown": 0.01, "p_significant_slowdown": 0.04, '
        '"p_moderate_slowdown": 0.05, "p_minor_slowdown": 0.10, '
        '"p_minor_speedup": 0.50, "p_significant_speedup": 0.20, '
        '"p_high_speedup": 0.05, "p_extreme_speedup": 0.05, '
        '"reasoning": "x"}'
    )
    both = Message(
        role="assistant",
        content="",
        tool_calls=[
            ToolCall(
                function=ToolCall.FunctionBody(
                    name="submit_kernel_runtime_estimate",
                    arguments=predict_args,
                )
            ),
            ToolCall(
                function=ToolCall.FunctionBody(
                    name="defer_to_real_evaluator",
                    arguments='{"reason": "noisy"}',
                )
            ),
        ],
    )
    env._renderer.parse_response = lambda response: (both, True)  # type: ignore[method-assign]

    result = asyncio.run(env.step(action=[1, 2, 3]))
    assert reward.last_outcome is None
    assert result.metrics["parsed"] == 0.0


def test_step_treats_unknown_tool_as_parse_failure() -> None:
    item = _item()
    reward = _FixedReward()
    env = _make_env(item, reward)
    asyncio.run(env.initial_observation())

    weird = Message(
        role="assistant",
        content="",
        tool_calls=[
            ToolCall(
                function=ToolCall.FunctionBody(
                    name="some_other_tool",
                    arguments="{}",
                )
            ),
        ],
    )
    env._renderer.parse_response = lambda response: (weird, True)  # type: ignore[method-assign]

    result = asyncio.run(env.step(action=[1, 2, 3]))
    assert reward.last_outcome is None
    assert result.metrics["parsed"] == 0.0


def test_step_treats_malformed_predict_args_as_parse_failure() -> None:
    item = _item()
    reward = _FixedReward()
    env = _make_env(item, reward)
    asyncio.run(env.initial_observation())

    bad = Message(
        role="assistant",
        content="",
        tool_calls=[
            ToolCall(
                function=ToolCall.FunctionBody(
                    name="submit_kernel_runtime_estimate",
                    arguments="not json",
                )
            ),
        ],
    )
    env._renderer.parse_response = lambda response: (bad, True)  # type: ignore[method-assign]

    result = asyncio.run(env.step(action=[1, 2, 3]))
    assert reward.last_outcome is None
    assert result.metrics["parsed"] == 0.0


def test_step_treats_malformed_defer_args_as_parse_failure() -> None:
    item = _item()
    reward = _FixedReward()
    env = _make_env(item, reward)
    asyncio.run(env.initial_observation())

    # `reason` field missing — DeferArguments enforces min_length=1.
    bad = Message(
        role="assistant",
        content="",
        tool_calls=[
            ToolCall(
                function=ToolCall.FunctionBody(
                    name="defer_to_real_evaluator",
                    arguments="{}",
                )
            ),
        ],
    )
    env._renderer.parse_response = lambda response: (bad, True)  # type: ignore[method-assign]

    result = asyncio.run(env.step(action=[1, 2, 3]))
    assert reward.last_outcome is None
    assert result.metrics["parsed"] == 0.0
