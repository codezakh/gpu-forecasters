"""Pack-generic tooling for building held-out kernel evaluation datasets.

See ``README.md`` for the design overview.
"""

from gpu_forecasters.eval_dataset_builder.v1.bin_filler import BinFiller
from gpu_forecasters.eval_dataset_builder.v1.domain import (
    BinFillRequest,
    BinFillResult,
    EvalDataset,
    EvalSet,
    EvalSetManifest,
    EvaluationProviderSpec,
    HarvestedKernelSource,
    KernelGenerationAttempt,
    KernelRuntimeComparison,
    MutationProviderSpec,
    NumKernelsForSpeedupBin,
    RequestForKernelInGoalSpeedupBin,
    RunSummary,
    SpeedupBand,
    speedup_band_for_bin,
)
from gpu_forecasters.eval_dataset_builder.v1.goal_conditioned_evaluation import (
    GoalConditionedEvaluationProvider,
    score_evaluation_against_target_bin,
)
from gpu_forecasters.eval_dataset_builder.v1.goal_conditioned_mutation.provider import (
    GoalConditionedMutationProvider,
    MutationError,
    build_render_context,
    render_prompt,
)
from gpu_forecasters.eval_dataset_builder.v1.orchestrator import (
    build_eval_dataset,
    fill_via_generation,
    harvest_into_eval_set,
    read_eval_dataset,
    write_eval_set,
)
from gpu_forecasters.eval_dataset_builder.v1.seed_selection import SelectedSeed, select_seed
from gpu_forecasters.eval_dataset_builder.v1.summary import (
    compute_run_summary_from_event_log,
    extract_in_target_kernels_from_event_log,
)
from gpu_forecasters.eval_dataset_builder.v1.v2_event_log_source import V2EventLogSource


__all__ = [
    "BinFillRequest",
    "BinFillResult",
    "BinFiller",
    "EvalDataset",
    "EvalSet",
    "EvalSetManifest",
    "EvaluationProviderSpec",
    "GoalConditionedEvaluationProvider",
    "GoalConditionedMutationProvider",
    "HarvestedKernelSource",
    "KernelGenerationAttempt",
    "KernelRuntimeComparison",
    "MutationError",
    "MutationProviderSpec",
    "NumKernelsForSpeedupBin",
    "RequestForKernelInGoalSpeedupBin",
    "RunSummary",
    "SelectedSeed",
    "SpeedupBand",
    "V2EventLogSource",
    "build_eval_dataset",
    "build_render_context",
    "compute_run_summary_from_event_log",
    "extract_in_target_kernels_from_event_log",
    "fill_via_generation",
    "harvest_into_eval_set",
    "read_eval_dataset",
    "render_prompt",
    "score_evaluation_against_target_bin",
    "select_seed",
    "speedup_band_for_bin",
    "write_eval_set",
]
