"""Configuration objects for the v3 surrogate-filtered search experiment harness.

One ``SurrogateSearchExperimentConfig`` parameterizes a per-pack run end
to end. The sub-configs name the four moving parts: PUCT search shape,
the kernel-mutating LLM, the speedup-forecasting surrogate, and the
Modal evaluator. Two configs that differ in any field are two
different experiments, per the project's "experiment config in git"
convention.

The surrogate sub-config is a discriminated union because real
experiments will compare a Tinker-hosted trained checkpoint against a
LiteLLM-hosted frontier model on the same pack. Encoding the provider
choice in the type makes the construction site in ``runner.py``
exhaustive over the variants and keeps invalid combinations (e.g. a
``model_path`` on a LiteLLM call) unrepresentable.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from arid_badger.gpu_mode_kernel.aggregation import AggregationMethod
from arid_badger.landscape_map.v2 import HardwareContext
from arid_badger.max_reward_puct.v3.config import SearchConfig


class MutatorConfig(BaseModel):
    """LLM that proposes kernel edits via the gpu-mode feedback prompt.

    ``max_tokens=None`` is correct for Gemini 3 Flash (which rejects a
    cap); Together-hosted gpt-oss requires an explicit cap. The
    request timeout bounds one litellm call before its internal retry.
    """

    model_config = ConfigDict(frozen=True)

    model_slug: str
    max_llm_concurrency: int
    request_timeout_s: float
    max_tokens: int | None
    temperature: float = 1.0
    num_retries: int = 4


class TinkerSurrogateConfig(BaseModel):
    """Speedup surrogate served by a Tinker SamplingClient.

    ``checkpoint_uri=None`` runs the bare base model (no RL training).
    The renderer selects the prompt template the model was trained
    against — the e0159 GRPO+Brier checkpoint and the bare gpt-oss-20b
    both use ``gpt_oss_medium_reasoning``.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["tinker"] = "tinker"
    base_model: str
    checkpoint_uri: str | None
    renderer_name: str
    temperature: float
    max_tokens: int
    max_retries: int


class LiteLlmSurrogateConfig(BaseModel):
    """Speedup surrogate served by a LiteLLM endpoint.

    Used for frontier API providers (Gemini, DeepSeek) and for
    Together-hosted open-weights baselines that we did not train.
    """

    model_config = ConfigDict(frozen=True)

    kind: Literal["litellm"] = "litellm"
    model_slug: str
    temperature: float
    max_tokens: int
    request_timeout_s: float
    max_retries: int


SurrogateConfig = Annotated[
    TinkerSurrogateConfig | LiteLlmSurrogateConfig,
    Field(discriminator="kind"),
]


class EvaluationConfig(BaseModel):
    """Modal-side knobs for the on-GPU evaluator."""

    model_config = ConfigDict(frozen=True)

    gpu: str
    aggregator: AggregationMethod
    max_in_flight: int


class SurrogateSearchExperimentConfig(BaseModel):
    """One experiment's full parameterization.

    The pack and observation type are *not* in this config because they
    are the subject of the experiment, not knobs. Two configs with
    different pack assignments would conflate "study different settings
    on one pack" with "study one setting on different packs"; the
    runner takes the pack as a separate argument so this distinction
    stays clean.

    ``hardware`` conditions both the Modal evaluator (via
    ``evaluation.gpu`` choosing the Modal GPU class) and the surrogate
    prompt (via the ``KernelRuntimeQuery`` it sees). Pinning a single
    ``HardwareContext`` here is what guarantees the two sides are
    talking about the same device.
    """

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    search: SearchConfig
    mutator: MutatorConfig
    surrogate: SurrogateConfig
    evaluation: EvaluationConfig
    hardware: HardwareContext
    num_runs: int = 1


# A100-80GB SXM4 specs — the device every paper run targets. Provided
# here so the per-pack experiment file can name the constant rather
# than re-declare the eight HardwareContext fields verbatim.
A100_80GB_SXM4_HARDWARE = HardwareContext(
    device_name="NVIDIA A100-SXM4-80GB",
    compute_capability=(8, 0),
    total_global_memory_gb=80.0,
    multiprocessor_count=108,
    max_threads_per_multiprocessor=2048,
    clock_rate_ghz=1.41,
    memory_clock_rate_ghz=1.512,
    memory_bus_width_bits=5120,
)


__all__ = [
    "A100_80GB_SXM4_HARDWARE",
    "EvaluationConfig",
    "LiteLlmSurrogateConfig",
    "MutatorConfig",
    "SurrogateConfig",
    "SurrogateSearchExperimentConfig",
    "TinkerSurrogateConfig",
]
