"""Pydantic config schemas for every runbook script.

JSON parsed via ``Model.model_validate_json``. Every field carries a
``Field(description=...)`` so the JSON Schema sidecar (emitted next to
each config dir) gives editors meaningful autocomplete and doc hints.
Two configs that differ in any field are two different reproductions.

Surrogate backends are a tagged union over ``kind``:

* ``litellm`` — frontier API providers (Gemini, DeepSeek, Together).
* ``tinker`` — bare ``openai/gpt-oss-20b`` via Tinker SamplingClient.
* ``tinker_lora`` — base + a LoRA adapter loaded from an HF model repo
  (e.g. ``codezakh/gpu-forecasters-gpt-oss-20b-correctness``).

The training reward is a separate tagged union over ``kind`` so the
three trained variants live as three small config files.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field


_FROZEN = ConfigDict(frozen=True, extra="forbid")


# ---------------------------------------------------------------------------
# Surrogate backends — used by 01, 03, 05 (scoring) and 04 (kernel search).
# ---------------------------------------------------------------------------


class LiteLlmBackendConfig(BaseModel):
    """Surrogate served by a LiteLLM endpoint.

    ``model_slug`` follows LiteLLM's prefixed convention
    (``gemini/...``, ``together_ai/openai/...``, ``deepseek/...``).
    ``max_tokens`` must leave headroom for reasoning models — gpt-oss
    exhausts its budget in the reasoning channel before emitting the
    tool call.
    """

    model_config = _FROZEN

    kind: Literal["litellm"] = "litellm"
    model_slug: str = Field(description="LiteLLM model slug (e.g. 'gemini/gemini-3-flash-preview').")
    temperature: float = Field(default=1.0, description="Sampling temperature.")
    max_tokens: int = Field(default=32000, description="Maximum tokens per completion.")
    request_timeout_s: float = Field(default=900.0, description="HTTP request timeout per call.")
    num_retries: int = Field(default=4, description="LiteLLM-internal retry count on transient failures.")


class DeepSeekBackendConfig(BaseModel):
    """DeepSeek surrogate via LiteLLM, kept distinct so the JSON config

    file can use a discoverable ``kind`` discriminator. Behaviour is
    identical to ``LiteLlmBackendConfig`` — the field shape and the
    underlying request path are the same; only the ``kind`` tag
    differs so the union resolves unambiguously.
    """

    model_config = _FROZEN

    kind: Literal["deepseek"] = "deepseek"
    model_slug: str = Field(description="LiteLLM model slug (e.g. 'deepseek/deepseek-chat').")
    temperature: float = Field(default=1.0, description="Sampling temperature.")
    max_tokens: int = Field(default=32000, description="Maximum tokens per completion.")
    request_timeout_s: float = Field(default=900.0, description="HTTP request timeout per call.")
    num_retries: int = Field(default=4, description="LiteLLM-internal retry count.")


class TinkerBackendConfig(BaseModel):
    """Bare base-model surrogate served by ``tinker.SamplingClient``.

    Used for the untrained gpt-oss-20b baseline arm. The renderer is
    the cookbook's ``gpt_oss_medium_reasoning`` — empirically the
    setting where gpt-oss-20b reliably emits the speedup-estimate tool
    call instead of answering in the ``final`` channel.
    """

    model_config = _FROZEN

    kind: Literal["tinker"] = "tinker"
    base_model: str = Field(default="openai/gpt-oss-20b", description="Tinker base model identifier.")
    renderer_name: str = Field(default="gpt_oss_medium_reasoning", description="Cookbook renderer key.")
    temperature: float = Field(default=1.0, description="Sampling temperature.")
    max_tokens: int = Field(default=16384, description="Maximum tokens per completion.")


BackendConfig = Annotated[
    LiteLlmBackendConfig | DeepSeekBackendConfig | TinkerBackendConfig,
    Field(discriminator="kind"),
]


# ---------------------------------------------------------------------------
# 01 / 05 — surrogate scoring on the canonical eval set / discovery pairs.
# ---------------------------------------------------------------------------


class BaselineScoringConfig(BaseModel):
    """Config for ``01_score_baseline.py``.

    Scores one surrogate (one config = one surrogate) across the
    requested packs of the canonical eval set, ``n_repeats`` times at
    the configured temperature. Output is one JSONL per (repeat, pack)
    plus a top-level summary.
    """

    model_config = _FROZEN

    surrogate_label: str = Field(description="Human-readable label written into the output and summaries.")
    backend: BackendConfig = Field(description="Surrogate provider configuration.")
    packs: tuple[str, ...] | None = Field(
        default=None,
        description="Eval-set packs to score. None means all six packs.",
    )
    n_repeats: int = Field(default=3, ge=1, description="Number of independent samples per row.")
    max_concurrency: int = Field(default=8, ge=1, description="Maximum concurrent LLM calls.")


class DiscoveryScoringConfig(BaseModel):
    """Config for ``05_score_discovery.py``.

    Same shape as ``BaselineScoringConfig`` but draws rows from the
    ``codezakh/gpu-forecasters-discovery-pairs`` dataset, where the
    anchor is the parent kernel and the candidate is the child. The
    surrogate forecasts the child's speedup relative to the parent.
    """

    model_config = _FROZEN

    surrogate_label: str = Field(description="Human-readable label written into the output.")
    backend: BackendConfig = Field(description="Surrogate provider configuration.")
    benchmark_families: tuple[Literal["gpu_mode", "kernelbench_l3"], ...] = Field(
        default=("gpu_mode", "kernelbench_l3"),
        description="Which discovery-pair families to score.",
    )
    n_repeats: int = Field(default=3, ge=1, description="Number of independent samples per pair.")
    max_concurrency: int = Field(default=8, ge=1, description="Maximum concurrent LLM calls.")


# ---------------------------------------------------------------------------
# 03 — trained-checkpoint scoring.
# ---------------------------------------------------------------------------


class TrainedScoringConfig(BaseModel):
    """Config for ``03_score_trained.py``.

    Describes only the scoring behaviour. The trained checkpoint
    itself is not in this config — it comes in as ``--training-artifact``
    pointing at the ``training_artifact.json`` produced by an upstream
    ``02_train_surrogate.py`` run, since Tinker checkpoint URIs are
    account-private and only meaningful to the Tinker account that
    produced them.
    """

    model_config = _FROZEN

    surrogate_label: str = Field(description="Human-readable label for the trained variant.")
    base_model: str = Field(default="openai/gpt-oss-20b", description="Tinker base model identifier.")
    renderer_name: str = Field(default="gpt_oss_medium_reasoning", description="Cookbook renderer key.")
    temperature: float = Field(default=1.0, description="Sampling temperature.")
    max_tokens: int = Field(default=16384, description="Maximum tokens per completion.")
    packs: tuple[str, ...] | None = Field(
        default=None,
        description="Eval-set packs to score. None means all six packs.",
    )
    n_repeats: int = Field(default=3, ge=1, description="Number of independent samples per row.")
    max_concurrency: int = Field(default=8, ge=1, description="Maximum concurrent LLM calls.")


# ---------------------------------------------------------------------------
# 02 — GRPO training.
# ---------------------------------------------------------------------------


class CorrectnessRewardConfig(BaseModel):
    """Binary correctness — ``r_total = 1[predicted_bin == true_bin]``."""

    model_config = _FROZEN

    kind: Literal["correctness"] = "correctness"


class CorrectnessBrierRewardConfig(BaseModel):
    """Correctness + Brier reward over the predicted distribution."""

    model_config = _FROZEN

    kind: Literal["correctness_brier"] = "correctness_brier"
    brier_weight: float = Field(default=1.0, description="Mixing weight applied to the Brier term.")


class CorrectnessCrpsRewardConfig(BaseModel):
    """Correctness + CRPS reward over the predicted distribution."""

    model_config = _FROZEN

    kind: Literal["correctness_crps"] = "correctness_crps"
    crps_weight: float = Field(default=1.0, description="Mixing weight applied to the CRPS term.")


RewardConfig = Annotated[
    CorrectnessRewardConfig | CorrectnessBrierRewardConfig | CorrectnessCrpsRewardConfig,
    Field(discriminator="kind"),
]


class TrainingRunConfig(BaseModel):
    """Config for ``02_train_surrogate.py``.

    Reproduces e0158 / e0159 / e0160 — the three trained surrogate
    variants in the paper. Input is the
    ``codezakh/gpu-forecasters-rl-training-pool`` dataset; output is a
    Tinker checkpoint URI persisted to disk as
    ``training_artifact.json``.
    """

    model_config = _FROZEN

    base_model: str = Field(default="openai/gpt-oss-20b", description="Base model id passed to Tinker.")
    renderer_name: str = Field(default="gpt_oss_medium_reasoning", description="Cookbook renderer key.")
    learning_rate: float = Field(default=4e-5, description="GRPO learning rate.")
    group_size: int = Field(default=8, ge=1, description="Samples drawn per training prompt (GRPO).")
    groups_per_batch: int = Field(default=8, ge=1, description="Distinct prompts per batch.")
    num_iters: int = Field(default=20, ge=1, description="Total GRPO iterations.")
    save_every: int = Field(default=5, ge=1, description="Checkpoint cadence (iterations).")
    max_tokens: int = Field(default=8192, ge=1, description="Maximum tokens per sample during rollout.")
    lora_rank: int = Field(default=32, ge=1, description="LoRA adapter rank.")
    temperature: float = Field(default=1.0, description="Rollout temperature.")
    reward: RewardConfig = Field(description="Reward function for GRPO.")
    training_packs: tuple[str, ...] | None = Field(
        default=None,
        description="Restrict training pool to these packs. None means all packs.",
    )


# ---------------------------------------------------------------------------
# 00 / 04 — PUCT search.
# ---------------------------------------------------------------------------


class HardwareSpec(BaseModel):
    """The GPU device the search targets.

    The default is the A100-80GB SXM4 that every paper run was on. The
    paper does include the L40S KernelBench-L3 runs (e0127 family); set
    ``gpu="L40S"`` for those.
    """

    model_config = _FROZEN

    gpu: Literal["A100-80GB", "H100", "B200", "L40S"] = Field(
        default="A100-80GB",
        description="Modal GPU class to dispatch evaluations to.",
    )


class StandardSearchMode(BaseModel):
    """Standard PUCT: every parent's ``samples_per_parent`` mutations are evaluated on GPU.

    No surrogate filter. This is the configuration the ``e00XX_..._puct``
    upstream searches used.
    """

    model_config = _FROZEN

    kind: Literal["standard"] = "standard"


class SurrogateFilteredSearchMode(BaseModel):
    """Surrogate-filtered PUCT (§4.4): ``samples_per_parent`` mutations forecast, top ``k_per_parent`` evaluated.

    The surrogate's inference is served by Tinker. If
    ``--surrogate-training-artifact`` is passed to
    ``04_kernel_search.py``, the script reads the trained checkpoint
    URI from there and uses it as the surrogate; otherwise the surrogate
    runs against the bare ``base_model``. Selection rule is
    expected-bin-index over the predicted distribution.
    """

    model_config = _FROZEN

    kind: Literal["surrogate_filtered"] = "surrogate_filtered"
    surrogate: TinkerBackendConfig = Field(
        description="Surrogate inference parameters (base model, renderer, temperature, max tokens).",
    )
    max_retries: int = Field(default=1, ge=0, description="Surrogate parse-error retries.")


SearchMode = Annotated[
    StandardSearchMode | SurrogateFilteredSearchMode,
    Field(discriminator="kind"),
]


class UpstreamPuctConfig(BaseModel):
    """Config for ``00_upstream_puct.py``.

    Reproduces one of the upstream PUCT searches that produced the raw
    data archive. ``pack`` selects which kernel pack (TriMul,
    cross-entropy, fp8_quant, etc.) the search runs against.
    """

    model_config = _FROZEN

    pack: Literal[
        "trimul",
        "cross_entropy",
        "gdn_chunk_fwd_h",
        "gdn_chunk_fwd_o",
        "gdn_recompute_w_u",
        "fp8_quant",
    ] = Field(description="GPU Mode kernel pack to search.")
    total_budget_steps: int = Field(default=40, ge=1, description="Total PUCT iterations.")
    batch_size: int = Field(default=2, ge=1, description="Parents selected per step.")
    samples_per_parent: int = Field(default=4, ge=1, description="Mutations dispatched per parent.")
    k_per_parent: int = Field(default=2, ge=1, description="Mutations promoted to paid evaluation per parent.")
    archive_capacity: int = Field(default=1000, ge=1, description="Maximum archived nodes.")
    c_puct: float = Field(default=1.0, description="PUCT exploration constant.")
    mutator_model_slug: str = Field(
        default="gemini/gemini-3-flash-preview",
        description="LLM that proposes kernel edits.",
    )
    max_llm_concurrency: int = Field(default=8, ge=1, description="Maximum concurrent mutator calls.")
    request_timeout_s: float = Field(default=900.0, description="Mutator request timeout.")
    hardware: HardwareSpec = Field(default_factory=HardwareSpec, description="Target GPU.")
    aggregator: Literal["geomean", "arith_mean", "min"] = Field(
        default="geomean",
        description="Per-case-speedup aggregator.",
    )


class KernelSearchConfig(BaseModel):
    """Config for ``04_kernel_search.py``.

    The §4.4 budget-matched comparison: same PUCT shape as the upstream
    searches, but ``mode`` toggles between the standard
    (no-surrogate) baseline and the surrogate-filtered variant whose
    paid budget is matched by ``k_per_parent < samples_per_parent``.
    """

    model_config = _FROZEN

    pack: Literal[
        "trimul",
        "cross_entropy",
        "gdn_chunk_fwd_h",
        "gdn_chunk_fwd_o",
        "gdn_recompute_w_u",
        "fp8_quant",
    ] = Field(description="GPU Mode kernel pack to search.")
    mode: SearchMode = Field(description="Standard or surrogate-filtered PUCT.")
    total_budget_steps: int = Field(default=40, ge=1, description="Total PUCT iterations.")
    batch_size: int = Field(default=2, ge=1, description="Parents selected per step.")
    samples_per_parent: int = Field(default=4, ge=1, description="Mutations dispatched per parent.")
    k_per_parent: int = Field(default=2, ge=1, description="Mutations promoted to paid evaluation per parent.")
    archive_capacity: int = Field(default=1000, ge=1, description="Maximum archived nodes.")
    c_puct: float = Field(default=1.0, description="PUCT exploration constant.")
    mutator_model_slug: str = Field(
        default="gemini/gemini-3-flash-preview",
        description="LLM that proposes kernel edits.",
    )
    max_llm_concurrency: int = Field(default=8, ge=1, description="Maximum concurrent mutator calls.")
    request_timeout_s: float = Field(default=900.0, description="Mutator request timeout.")
    hardware: HardwareSpec = Field(default_factory=HardwareSpec, description="Target GPU.")
    aggregator: Literal["geomean", "arith_mean", "min"] = Field(
        default="geomean",
        description="Per-case-speedup aggregator.",
    )


# ---------------------------------------------------------------------------
# 06 — figures.
# ---------------------------------------------------------------------------


FigureKey = Literal[
    "tab1_headline",
    "fig1_calibration",
    "fig2_forecast_error",
    "fig3_per_pack",
    "fig4_kernel_search",
    "fig5_discovery_precision_recall",
    "throughput",
]


class FigureConfig(BaseModel):
    """Config for ``06_make_figures.py``.

    Each figure is regenerated from the public HF artifacts: scored
    eval-set predictions for §4.3 and §4.4, discovery-pair scoring for
    §4.5, and a tiny pre-shipped fixture for the throughput figure.
    """

    model_config = _FROZEN

    figures: tuple[FigureKey, ...] = Field(
        default=(
            "tab1_headline",
            "fig1_calibration",
            "fig2_forecast_error",
            "fig3_per_pack",
            "fig4_kernel_search",
            "fig5_discovery_precision_recall",
            "throughput",
        ),
        description="Subset of figures to render.",
    )
    surrogates: tuple[str, ...] = Field(
        default=(
            "gemini3_flash",
            "gpt_oss_120b",
            "gpt_oss_20b_untrained",
            "deepseek_v4",
            "trained_correctness",
            "trained_correctness_brier",
            "trained_correctness_crps",
        ),
        description="HF eval-set-predictions config names to include.",
    )


__all__ = [
    "BackendConfig",
    "BaselineScoringConfig",
    "CorrectnessBrierRewardConfig",
    "CorrectnessCrpsRewardConfig",
    "CorrectnessRewardConfig",
    "DeepSeekBackendConfig",
    "DiscoveryScoringConfig",
    "FigureConfig",
    "FigureKey",
    "HardwareSpec",
    "KernelSearchConfig",
    "LiteLlmBackendConfig",
    "RewardConfig",
    "SearchMode",
    "StandardSearchMode",
    "SurrogateFilteredSearchMode",
    "TinkerBackendConfig",
    "TrainedScoringConfig",
    "TrainingRunConfig",
    "UpstreamPuctConfig",
]
