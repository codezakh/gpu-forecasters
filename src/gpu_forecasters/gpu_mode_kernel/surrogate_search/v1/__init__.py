"""v1 surrogate-filtered search harness for gpu-mode kernel packs.

Top-level surface mirrors the v2 ``experiment_helper``: callers build
one ``SurrogateSearchExperimentConfig`` and pass it to
``run_pack_experiment`` along with the pack runtime and case-speedup
type. See ``README.md`` for the version's purpose.
"""

from gpu_forecasters.gpu_mode_kernel.surrogate_search.v1.config import (
    A100_80GB_SXM4_HARDWARE,
    EvaluationConfig,
    LiteLlmSurrogateConfig,
    MutatorConfig,
    SurrogateConfig,
    SurrogateSearchExperimentConfig,
    TinkerSurrogateConfig,
)
from gpu_forecasters.gpu_mode_kernel.surrogate_search.v1.runner import (
    RunSummary,
    load_run_summaries,
    run_pack_experiment,
)


__all__ = [
    "A100_80GB_SXM4_HARDWARE",
    "EvaluationConfig",
    "LiteLlmSurrogateConfig",
    "MutatorConfig",
    "RunSummary",
    "SurrogateConfig",
    "SurrogateSearchExperimentConfig",
    "TinkerSurrogateConfig",
    "load_run_summaries",
    "run_pack_experiment",
]
