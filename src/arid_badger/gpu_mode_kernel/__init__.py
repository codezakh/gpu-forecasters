"""Generic infrastructure for evolutionary search over gpu-mode-style kernels.

Each kernel-specific package (``arid_badger.trimul``,
``arid_badger.causal_conv1d``) used to mirror the same ~10-file layout:
``cases.py``, ``comparison.py``, ``core.py``, ``reference.py``,
``scoring.py``, ``modal_scoring.py``, ``seed_kernel.py``, plus four
providers under ``hill_climbing/`` and ``max_reward_puct.v2/``. ~95% of
that was kernel-agnostic.

This package replaces that pattern: kernel-specific behavior collapses
into a single ``KernelPack`` value object (see ``kernel_pack.py``);
the scoring pipeline, Modal harness, and v2 providers consume the
pack generically.

Adding a new gpu-mode/reference-kernels problem becomes a single file
under ``arid_badger.gpu_mode_kernel.packs.<name>`` declaring a
``<NAME>_PACK: KernelPack[...]`` constant.
"""

from arid_badger.gpu_mode_kernel.aggregation import (
    AggregationMethod,
    AggregationResult,
    aggregate_outcomes,
    aggregate_speedups,
)
from arid_badger.gpu_mode_kernel.core import (
    CaseSpeedupBase,
    CompileFailedFeedback,
    GpuModeKernelObservation,
    IncorrectFeedback,
    InfrastructureFailureFeedback,
    KernelExecResult,
    KernelExecutionFeedback,
    KernelFailureFeedback,
    ObservationFeedback,
    RuntimeErrorFeedback,
    Stats,
    SuccessFeedback,
    failure_feedback_from_exec_result,
)
from arid_badger.gpu_mode_kernel.kernel_pack import KernelPack


__all__ = [
    "AggregationMethod",
    "AggregationResult",
    "CaseSpeedupBase",
    "CompileFailedFeedback",
    "GpuModeKernelObservation",
    "IncorrectFeedback",
    "InfrastructureFailureFeedback",
    "KernelExecResult",
    "KernelExecutionFeedback",
    "KernelFailureFeedback",
    "KernelPack",
    "ObservationFeedback",
    "RuntimeErrorFeedback",
    "Stats",
    "SuccessFeedback",
    "aggregate_outcomes",
    "aggregate_speedups",
    "failure_feedback_from_exec_result",
]
