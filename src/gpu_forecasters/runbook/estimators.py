"""Construct ``AsyncSpeedupEstimator`` instances from runbook backend configs.

Single dispatch point so every scoring script (01, 03, 05) takes the
same path from JSON to a live estimator. The trained-LoRA variant
goes through ``TinkerSamplingClientEstimator`` with ``model_path`` set
to the HF adapter repo ID — Tinker resolves the LoRA from there.
"""

from __future__ import annotations

from gpu_forecasters.landscape_map.v2 import (
    AsyncSpeedupEstimator,
    LlmSpeedupEstimator,
    TinkerSamplingClientEstimator,
)
from gpu_forecasters.runbook.configs import (
    BackendConfig,
    DeepSeekBackendConfig,
    LiteLlmBackendConfig,
    TinkerBackendConfig,
)


def build_estimator_from_backend(backend: BackendConfig) -> AsyncSpeedupEstimator:
    """Return an ``AsyncSpeedupEstimator`` instance for ``backend``.

    Both the LiteLLM and Tinker estimators are safe to share across
    concurrent calls. Trained-checkpoint scoring goes through
    :func:`build_trained_estimator` instead — it takes a Tinker
    checkpoint URI that comes from an upstream training artifact.
    """
    if isinstance(backend, (LiteLlmBackendConfig, DeepSeekBackendConfig)):
        return LlmSpeedupEstimator(
            model_slug=backend.model_slug,
            temperature=backend.temperature,
            max_tokens=backend.max_tokens,
            request_timeout_s=backend.request_timeout_s,
            num_retries=backend.num_retries,
        )
    if isinstance(backend, TinkerBackendConfig):
        return TinkerSamplingClientEstimator(
            base_model=backend.base_model,
            renderer_name=backend.renderer_name,
            temperature=backend.temperature,
            max_tokens=backend.max_tokens,
        )
    raise TypeError(f"unknown backend kind: {backend!r}")


def build_trained_estimator(
    *,
    base_model: str,
    checkpoint_uri: str,
    renderer_name: str,
    temperature: float,
    max_tokens: int,
) -> AsyncSpeedupEstimator:
    """Construct a Tinker SamplingClient pinned to a trained checkpoint URI.

    ``checkpoint_uri`` must be a ``tinker://...`` URI produced by
    ``02_train_surrogate.py`` against the caller's own Tinker account
    — checkpoint URIs are account-private and only meaningful inside
    the account that produced them.
    """
    return TinkerSamplingClientEstimator(
        base_model=base_model,
        model_path=checkpoint_uri,
        renderer_name=renderer_name,
        temperature=temperature,
        max_tokens=max_tokens,
    )


__all__ = ["build_estimator_from_backend", "build_trained_estimator"]
