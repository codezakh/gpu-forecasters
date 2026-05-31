"""Single-turn cookbook ``Env`` for RL training of v2 surrogates.

Mirrors the inference path used by :class:`TinkerSamplingClientEstimator`
exactly so a checkpoint trained here samples in-distribution at eval
time:

  1. Build the conversation prefix with
     :func:`Renderer.create_conversation_prefix_with_tools`, passing the
     v2 :func:`cookbook_tool_spec` and the v2 :func:`render_system_prompt`.
  2. Append the user message produced by :func:`render_user_prompt`.
  3. Tokenize with :meth:`Renderer.build_generation_prompt`.
  4. Stop on :meth:`Renderer.get_stop_sequences` (gpt-oss: ``[<|return|>,
     <|call|>]``).
  5. On ``step``: call :meth:`Renderer.parse_response` on the action
     tokens, pick the ``submit_kernel_runtime_estimate`` tool call,
     and feed its arguments to :func:`parse_tool_call_args`.

The reward function is a constructor argument — we don't pick a
calibration term here; experiments do that. The env only knows how
to run one rollout, parse it, and ask a :class:`RewardFunction` to
score the parsed estimate against the truth bin.
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

from .domain import (
    HardwareContext,
    KernelImplementation,
    KernelRuntimeEstimate,
    KernelRuntimeQuery,
    KernelTaskInfo,
    SpeedupBin,
)
from .parsing import EstimatorParseError, parse_tool_call_args
from .prompt_rendering import render_system_prompt, render_user_prompt
from .tool_spec import TOOL_NAME, cookbook_tool_spec


# ---------------------------------------------------------------------------
# Training row
# ---------------------------------------------------------------------------


class LabeledKernelItem(BaseModel, frozen=True):
    """One labeled training row: anchor + candidate + truth + provenance.

    The query (anchor / candidate / hardware) is what the surrogate
    sees; the speedup is the supervision target; ``pack_name`` /
    ``source_id`` / ``anchor_source`` are provenance for logtree and
    metrics.

    A pack-unaware ``op_name`` (``"gpu_kernel"``) is the convention
    inherited from e0117/e0121 — keep prompts identical between training
    and eval so a trained checkpoint isn't out-of-distribution at
    scoring time.
    """

    pack_name: str
    anchor_source: str
    anchor_code: str
    candidate_code: str
    speedup_geomean: float
    hardware: HardwareContext
    source_id: str

    @property
    def true_bin(self) -> SpeedupBin:
        """Map ``speedup_geomean`` to its 1..8 bin via :meth:`SpeedupBin.from_speedup`."""
        return SpeedupBin.from_speedup(self.speedup_geomean)

    def to_query(self) -> KernelRuntimeQuery:
        """Build the v2 :class:`KernelRuntimeQuery` for prompt rendering."""
        return KernelRuntimeQuery(
            task=KernelTaskInfo(op_name="gpu_kernel", level_id=0, task_id=0),
            reference=KernelImplementation(
                kernel_name="reference",
                code=self.anchor_code,
                runtime_ms=None,
            ),
            candidate=KernelImplementation(
                kernel_name="candidate",
                code=self.candidate_code,
                runtime_ms=None,
            ),
            hardware=self.hardware,
        )


# ---------------------------------------------------------------------------
# Reward protocol
# ---------------------------------------------------------------------------


class RewardComponents(BaseModel, frozen=True):
    """Three numbers logged per rollout.

    ``total`` is the value the env returns to GRPO as the rollout's
    reward; ``distance`` and ``calibration`` are persisted to Tinker
    metrics so the two terms' evolution can be plotted from
    ``metrics.jsonl`` without re-deriving them after the fact.
    """

    distance: float
    calibration: float
    total: float


class RewardFunction(Protocol):
    """Maps (parsed estimate or parse failure, truth bin) to a triple.

    Experiments implement this protocol with a calibration term of
    their choice (NLL on the true bin, peak-confidence Brier, ...).
    The env doesn't know or care which.
    """

    def reward(
        self,
        estimate: KernelRuntimeEstimate | None,
        true_bin: SpeedupBin,
    ) -> RewardComponents: ...


# ---------------------------------------------------------------------------
# The env
# ---------------------------------------------------------------------------


def _render_messages(item: LabeledKernelItem) -> list[renderers.Message]:
    """User message that goes after the renderer's tool prefix."""
    return [
        {"role": "user", "content": render_user_prompt(item.to_query())},
    ]


def _parse_tool_call_or_none(
    message: renderers.Message,
) -> KernelRuntimeEstimate | None:
    """Extract the v2 estimate from a parsed assistant message, or None on failure."""
    tool_calls = list(message.get("tool_calls") or [])
    if not tool_calls:
        return None
    target = next(
        (tc for tc in tool_calls if tc.function.name == TOOL_NAME),
        None,
    )
    if target is None:
        return None
    try:
        return parse_tool_call_args(target.function.arguments)
    except EstimatorParseError:
        return None


class KernelBinPredictionEnv(Env):
    """Single-turn env: one prompt, one tool call, one reward.

    Constructed by an :class:`EnvGroupBuilder` once per rollout in a
    GRPO group. Re-constructed (no ``reset``) every time the trainer
    asks for a new env.
    """

    _item: LabeledKernelItem
    _renderer: renderers.Renderer
    _reward_fn: RewardFunction
    _true_bin: SpeedupBin

    def __init__(
        self,
        item: LabeledKernelItem,
        renderer: renderers.Renderer,
        reward_fn: RewardFunction,
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
            tools=[cookbook_tool_spec()],  # pyright: ignore[reportArgumentType]
            system_prompt=render_system_prompt(),
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
        estimate = _parse_tool_call_or_none(message) if parse_success else None
        components = self._reward_fn.reward(estimate, self._true_bin)
        predicted_bin = (
            int(estimate.predicted_bin) if estimate is not None else -1
        )

        with logtree.scope_header("Prompt"):
            prefix = self._renderer.create_conversation_prefix_with_tools(
                tools=[cookbook_tool_spec()],  # pyright: ignore[reportArgumentType]
                system_prompt=render_system_prompt(),
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
                    "predicted_bin": predicted_bin,
                    "parsed": estimate is not None,
                    "r_distance": f"{components.distance:.3f}",
                    "r_calibration": f"{components.calibration:.3f}",
                    "r_total": f"{components.total:.3f}",
                    "source_id": self._item.source_id,
                },
                caption="Reward components",
            )

        metrics: Metrics = {
            "parsed": float(estimate is not None),
            "exact_match": float(
                estimate is not None and estimate.predicted_bin == self._true_bin
            ),
            "off_by_one_or_less": float(
                estimate is not None
                and abs(int(estimate.predicted_bin) - int(self._true_bin)) <= 1
            ),
            "r_distance": components.distance,
            "r_calibration": components.calibration,
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
    "KernelBinPredictionEnv",
    "LabeledKernelItem",
    "RewardComponents",
    "RewardFunction",
]
