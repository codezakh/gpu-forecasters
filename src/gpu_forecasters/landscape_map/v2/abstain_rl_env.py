"""Single-turn cookbook ``Env`` for RL training of an *abstain-or-forecast* v2 surrogate.

Mirrors :class:`KernelBinPredictionEnv` but registers the two-tool
abstain surface (predict + defer) on the conversation prefix and
renders the abstain prompt:

  1. Build the conversation prefix with
     :func:`Renderer.create_conversation_prefix_with_tools`, passing
     ``both_cookbook_tool_specs()`` and
     :func:`render_abstain_system_prompt`.
  2. Append the user message produced by
     :func:`render_abstain_user_prompt`.
  3. Tokenize and sample to a stop sequence (gpt-oss:
     ``[<|return|>, <|call|>]``).
  4. On ``step``, call :meth:`Renderer.parse_response` on the action
     tokens. The model's assistant message must contain *exactly one*
     tool call, either ``submit_kernel_runtime_estimate`` (yielding a
     :class:`Forecast`) or ``defer_to_real_evaluator`` (yielding a
     :class:`Deferral`). Anything else is a parse failure (the env
     hands ``None`` to the reward function).

The reward shape lives in the experiment, not the library: the env
takes an :class:`AbstainRewardFunction` as a constructor argument
and only knows how to call it with the parsed
:class:`PredictOrDefer` outcome (or ``None``) and the truth bin.
"""

from __future__ import annotations

from typing import Protocol

import tinker
from pydantic import BaseModel
from tinker_cookbook import renderers
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.rl.types import (
    Action,
    ActionExtra,
    Env,
    Metrics,
    Observation,
    StepResult,
)
from tinker_cookbook.utils import logtree
from tinker_cookbook.utils.logtree_formatters import ConversationFormatter

from .abstain_outcome import Deferral, Forecast, PredictOrDefer
from .abstain_prompt_rendering import (
    render_abstain_system_prompt,
    render_abstain_user_prompt,
)
from .abstain_tool_spec import (
    DEFER_TOOL_NAME,
    DeferArguments,
    PREDICT_TOOL_NAME,
    both_cookbook_tool_specs,
)
from .domain import SpeedupBin
from .parsing import EstimatorParseError, parse_tool_call_args
from .rl_env import LabeledKernelItem


# ---------------------------------------------------------------------------
# Reward protocol
# ---------------------------------------------------------------------------


class AbstainRewardComponents(BaseModel, frozen=True):
    """Three numbers logged per rollout.

    ``total`` is the value the env returns to GRPO as the rollout's
    reward; ``correctness`` is the bin-level correctness term
    (``1`` iff the model forecasted *and* matched the truth bin); the
    ``abstention`` slot carries the abstain-or-forecast portion of
    the reward (a function of ``β`` / ``γ`` plus the correctness
    indicator). Persisting both lets the experiment plot how the
    abstain rate and the conditional accuracy evolve independently.
    """

    correctness: float
    abstention: float
    total: float


class AbstainRewardFunction(Protocol):
    """Maps (parsed outcome or parse failure, truth bin) to a triple.

    Experiments implement this protocol with whichever combination of
    abstain / forecast incentives they want; the env doesn't pick
    coefficients.
    """

    def reward(
        self,
        outcome: PredictOrDefer | None,
        true_bin: SpeedupBin,
    ) -> AbstainRewardComponents: ...


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------


def _parse_predict_or_defer_or_none(
    message: renderers.Message,
) -> PredictOrDefer | None:
    """Extract a :class:`Forecast` or :class:`Deferral` from a parsed assistant message.

    The contract is exactly one tool call, with name in
    ``{PREDICT_TOOL_NAME, DEFER_TOOL_NAME}``. Zero, multiple, or
    unknown tool calls — and any per-arm validation failure — return
    ``None`` so the reward function can treat it as a parse failure.
    """
    tool_calls = list(message.get("tool_calls") or [])
    if len(tool_calls) != 1:
        return None
    call = tool_calls[0]
    name = call.function.name
    if name == PREDICT_TOOL_NAME:
        try:
            estimate = parse_tool_call_args(call.function.arguments)
        except EstimatorParseError:
            return None
        return Forecast(estimate=estimate)
    if name == DEFER_TOOL_NAME:
        try:
            args = DeferArguments.model_validate_json(call.function.arguments)
        except Exception:
            return None
        return Deferral(reason=args.reason)
    return None


# ---------------------------------------------------------------------------
# The env
# ---------------------------------------------------------------------------


def _render_messages(item: LabeledKernelItem) -> list[renderers.Message]:
    """User message that goes after the renderer's tool prefix."""
    return [
        {"role": "user", "content": render_abstain_user_prompt(item.to_query())},
    ]


def _outcome_diagnostics(
    outcome: PredictOrDefer | None, true_bin: SpeedupBin
) -> dict[str, object]:
    """Per-rollout diagnostics surfaced to logtree and metrics."""
    if outcome is None:
        return {
            "kind": "parse_failure",
            "predicted_bin": -1,
            "deferred": False,
            "exact_match": False,
            "off_by_one_or_less": False,
        }
    if outcome.kind == "defer":
        return {
            "kind": "defer",
            "predicted_bin": -1,
            "deferred": True,
            "exact_match": False,
            "off_by_one_or_less": False,
            "defer_reason": outcome.reason,
        }
    estimate = outcome.estimate
    distance = abs(int(estimate.predicted_bin) - int(true_bin))
    return {
        "kind": "predict",
        "predicted_bin": int(estimate.predicted_bin),
        "deferred": False,
        "exact_match": estimate.predicted_bin == true_bin,
        "off_by_one_or_less": distance <= 1,
    }


class KernelBinPredictionAbstainEnv(Env):
    """Single-turn env: one prompt, one tool call (predict or defer), one reward.

    Constructed by an :class:`EnvGroupBuilder` once per rollout in a
    GRPO group. Re-constructed every time the trainer asks for a new
    env (no ``reset``).
    """

    _item: LabeledKernelItem
    _renderer: renderers.Renderer
    _reward_fn: AbstainRewardFunction
    _true_bin: SpeedupBin

    def __init__(
        self,
        item: LabeledKernelItem,
        renderer: renderers.Renderer,
        reward_fn: AbstainRewardFunction,
    ) -> None:
        self._item = item
        self._renderer = renderer
        self._reward_fn = reward_fn
        self._true_bin = item.true_bin

    @property
    def stop_condition(self) -> StopCondition:
        return self._renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        prefix = self._renderer.create_conversation_prefix_with_tools(
            tools=both_cookbook_tool_specs(),  # pyright: ignore[reportArgumentType]
            system_prompt=render_abstain_system_prompt(),
        )
        messages = list(prefix) + _render_messages(self._item)
        return (
            self._renderer.build_generation_prompt(messages),
            self.stop_condition,
        )

    async def step(
        self, action: Action, *, extra: ActionExtra | None = None
    ) -> StepResult:
        del extra
        message, parse_success = self._renderer.parse_response(action)
        outcome = (
            _parse_predict_or_defer_or_none(message) if parse_success else None
        )
        components = self._reward_fn.reward(outcome, self._true_bin)
        diag = _outcome_diagnostics(outcome, self._true_bin)

        with logtree.scope_header("Prompt"):
            prefix = self._renderer.create_conversation_prefix_with_tools(
                tools=both_cookbook_tool_specs(),  # pyright: ignore[reportArgumentType]
                system_prompt=render_abstain_system_prompt(),
            )
            logtree.log_formatter(
                ConversationFormatter(
                    messages=list(prefix) + _render_messages(self._item)
                )
            )
        with logtree.scope_header("Policy Response"):
            logtree.log_formatter(ConversationFormatter(messages=[message]))
        with logtree.scope_header("Reward"):
            logtree.table_from_dict(
                {
                    "pack": self._item.pack_name,
                    "anchor_source": self._item.anchor_source,
                    "speedup_geomean": f"{self._item.speedup_geomean:.4f}",
                    "true_bin": int(self._true_bin),
                    "outcome_kind": str(diag["kind"]),
                    "predicted_bin": diag["predicted_bin"],
                    "deferred": diag["deferred"],
                    "r_correctness": f"{components.correctness:.3f}",
                    "r_abstention": f"{components.abstention:.3f}",
                    "r_total": f"{components.total:.3f}",
                    "source_id": self._item.source_id,
                },
                caption="Reward components",
            )

        metrics: Metrics = {
            "parsed": float(outcome is not None),
            "deferred": float(bool(diag["deferred"])),
            "forecasted": float(
                outcome is not None and not bool(diag["deferred"])
            ),
            "exact_match": float(bool(diag["exact_match"])),
            "off_by_one_or_less": float(bool(diag["off_by_one_or_less"])),
            "r_correctness": components.correctness,
            "r_abstention": components.abstention,
            "r_total": components.total,
        }

        return StepResult(
            reward=components.total,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics=metrics,
        )


__all__ = [
    "AbstainRewardComponents",
    "AbstainRewardFunction",
    "KernelBinPredictionAbstainEnv",
]
