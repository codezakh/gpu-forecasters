"""Landscape map model v2 — tool-calling LLM surrogate with numerical uncertainty.

The v2 surrogate emits its prediction via a *single tool call* whose
arguments are a probability distribution over success bins 1..8 (a
true simplex), rather than v1's free-form text + Likert confidences.
The same prompt structure (bin table, ten-factor analysis guide,
hardware-context table) is preserved.

Public surface:

  - :class:`KernelRuntimeQuery` / :class:`KernelRuntimeEstimate` —
    domain types.
  - :class:`SpeedupEstimator` / :class:`AsyncSpeedupEstimator` —
    component protocols.
  - :class:`LlmSpeedupEstimator` — concrete estimator over LiteLLM.
  - :class:`TinkerSamplingClientEstimator` — concrete estimator over
    ``tinker.SamplingClient`` + cookbook ``GptOssRenderer``.
  - :class:`StubEstimator` — fixed-bin estimator for tests.
  - :func:`render_system_prompt` / :func:`render_user_prompt` — Jinja
    template renderers used by the estimators (re-exported so RL
    ``Env`` code can drive the same prompt without going through an
    estimator).
  - :func:`parse_tool_call_args` — JSON-arguments → domain estimate
    parser, used by both estimators *and* by RL ``Env.step`` after
    ``renderer.parse_response``.
  - :data:`TOOL_NAME` / :func:`openai_tool_spec` /
    :func:`cookbook_tool_spec` — tool definitions.
"""

from arid_badger.landscape_map.v2.abstain_outcome import (
    Deferral,
    Forecast,
    PredictOrDefer,
)
from arid_badger.landscape_map.v2.abstain_rl_env import (
    AbstainRewardComponents,
    AbstainRewardFunction,
    KernelBinPredictionAbstainEnv,
)
from arid_badger.landscape_map.v2.domain import (
    SUCCESS_BINS,
    AsyncSpeedupEstimator,
    HardwareContext,
    KernelImplementation,
    KernelRuntimeEstimate,
    KernelRuntimeQuery,
    KernelTaskInfo,
    LlmCallUsage,
    SpeedupBin,
    SpeedupEstimator,
    renormalize,
)
from arid_badger.landscape_map.v2.litellm_estimator import LlmSpeedupEstimator
from arid_badger.landscape_map.v2.parsing import (
    EstimatorParseError,
    parse_tool_call_args,
)
from arid_badger.landscape_map.v2.prompt_rendering import (
    render_system_prompt,
    render_user_prompt,
)
from arid_badger.landscape_map.v2.rl_env import (
    KernelBinPredictionEnv,
    LabeledKernelItem,
    RewardComponents,
    RewardFunction,
)
from arid_badger.landscape_map.v2.stub_estimator import StubEstimator
from arid_badger.landscape_map.v2.tinker_abstain_estimator import (
    AsyncAbstainSpeedupEstimator,
    TinkerSamplingClientAbstainingEstimator,
)
from arid_badger.landscape_map.v2.tinker_sampling_estimator import (
    TinkerSamplingClientEstimator,
)
from arid_badger.landscape_map.v2.tool_spec import (
    TOOL_DESCRIPTION,
    TOOL_NAME,
    SubmitEstimateArguments,
    cookbook_tool_spec,
    openai_tool_spec,
    parameters_schema,
)


__all__ = [
    # domain
    "AsyncSpeedupEstimator",
    "EstimatorParseError",
    "HardwareContext",
    "KernelImplementation",
    "KernelRuntimeEstimate",
    "KernelRuntimeQuery",
    "KernelTaskInfo",
    "LlmCallUsage",
    "SUCCESS_BINS",
    "SpeedupBin",
    "SpeedupEstimator",
    "renormalize",
    # tool spec
    "SubmitEstimateArguments",
    "TOOL_DESCRIPTION",
    "TOOL_NAME",
    "cookbook_tool_spec",
    "openai_tool_spec",
    "parameters_schema",
    # parsing + prompts
    "parse_tool_call_args",
    "render_system_prompt",
    "render_user_prompt",
    # RL env
    "KernelBinPredictionEnv",
    "LabeledKernelItem",
    "RewardComponents",
    "RewardFunction",
    # abstain outcome + env
    "AbstainRewardComponents",
    "AbstainRewardFunction",
    "Deferral",
    "Forecast",
    "KernelBinPredictionAbstainEnv",
    "PredictOrDefer",
    # concretes
    "AsyncAbstainSpeedupEstimator",
    "LlmSpeedupEstimator",
    "StubEstimator",
    "TinkerSamplingClientAbstainingEstimator",
    "TinkerSamplingClientEstimator",
]
