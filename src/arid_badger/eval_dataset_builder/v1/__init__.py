"""Pack-generic tooling for building held-out kernel evaluation datasets.

See ``README.md`` for the design overview.
"""

from arid_badger.eval_dataset_builder.v1.bin_filler import BinFiller
from arid_badger.eval_dataset_builder.v1.domain import (
    BinFillRequest,
    BinFillResult,
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
from arid_badger.eval_dataset_builder.v1.goal_conditioned_evaluation import (
    GoalConditionedEvaluationProvider,
    score_evaluation_against_target_bin,
)
from arid_badger.eval_dataset_builder.v1.goal_conditioned_mutation.provider import (
    GoalConditionedMutationProvider,
    MutationError,
    build_render_context,
    render_prompt,
)
from arid_badger.eval_dataset_builder.v1.orchestrator import (
    build_eval_dataset,
    fill_via_generation,
    harvest_into_eval_set,
    write_eval_set,
)
from arid_badger.eval_dataset_builder.v1.seed_selection import SelectedSeed, select_seed
from arid_badger.eval_dataset_builder.v1.summary import (
    compute_run_summary_from_event_log,
    extract_in_target_kernels_from_event_log,
)


__all__ = [
    "BinFillRequest",
    "BinFillResult",
    "BinFiller",
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
    "build_eval_dataset",
    "build_render_context",
    "compute_run_summary_from_event_log",
    "extract_in_target_kernels_from_event_log",
    "fill_via_generation",
    "harvest_into_eval_set",
    "render_prompt",
    "score_evaluation_against_target_bin",
    "select_seed",
    "speedup_band_for_bin",
    "write_eval_set",
]
