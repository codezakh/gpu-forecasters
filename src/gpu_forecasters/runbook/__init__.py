"""Public-release runbook support library.

Shared schema, dataset loaders, output writers, and debug-mode
overrides for the numbered scripts under ``15-arid-badger/runbook/``.
Everything a reader needs to reproduce a paper figure or table cell
flows through this package.

The runbook scripts themselves are thin entry points: they parse a
JSON config into one of the Pydantic models defined here, optionally
apply :func:`apply_debug_overrides`, then dispatch to existing library
code (``gpu_forecasters.landscape_map.v2``,
``gpu_forecasters.gpu_mode_kernel.surrogate_search.v1``, etc.).
"""

from gpu_forecasters.runbook.configs import (
    BaselineScoringConfig,
    BackendConfig,
    DeepSeekBackendConfig,
    DiscoveryScoringConfig,
    FigureConfig,
    KernelSearchConfig,
    LiteLlmBackendConfig,
    StandardSearchMode,
    SurrogateFilteredSearchMode,
    TinkerBackendConfig,
    TrainedScoringConfig,
    TrainingRunConfig,
    UpstreamPuctConfig,
)
from gpu_forecasters.runbook.datasets import (
    HF_DISCOVERY_PAIRS,
    HF_EVAL_SET,
    HF_EVAL_SET_PREDICTIONS,
    HF_LORA_REPOS,
    HF_PUCT_SEARCH_EVENTS,
    HF_RL_TRAINING_POOL,
    load_canonical_eval_set,
    load_discovery_pairs,
    load_rl_training_pool,
)
from gpu_forecasters.runbook.debug import apply_debug_overrides
from gpu_forecasters.runbook.estimators import build_estimator_from_backend


__all__ = [
    "BackendConfig",
    "BaselineScoringConfig",
    "DeepSeekBackendConfig",
    "DiscoveryScoringConfig",
    "FigureConfig",
    "HF_DISCOVERY_PAIRS",
    "HF_EVAL_SET",
    "HF_EVAL_SET_PREDICTIONS",
    "HF_LORA_REPOS",
    "HF_PUCT_SEARCH_EVENTS",
    "HF_RL_TRAINING_POOL",
    "KernelSearchConfig",
    "LiteLlmBackendConfig",
    "StandardSearchMode",
    "SurrogateFilteredSearchMode",
    "TinkerBackendConfig",
    "TrainedScoringConfig",
    "TrainingRunConfig",
    "UpstreamPuctConfig",
    "apply_debug_overrides",
    "build_estimator_from_backend",
    "load_canonical_eval_set",
    "load_discovery_pairs",
    "load_rl_training_pool",
]
