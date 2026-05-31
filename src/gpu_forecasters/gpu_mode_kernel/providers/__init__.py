"""Generic v2 search providers for gpu-mode-style kernels.

These satisfy ``AsyncEvaluationProvider[GpuModeKernelObservation[CaseSpeedupT]]``
and ``AsyncMutationProvider[GpuModeKernelObservation[CaseSpeedupT]]`` via
the ``KernelPack`` seam, without per-kernel duplication.
"""

from arid_badger.gpu_mode_kernel.providers.v2_feedback_mutation import (
    GpuModeKernelFeedbackMutationProvider,
    MutationError,
)
from arid_badger.gpu_mode_kernel.providers.v2_modal_scoring import (
    GpuModeKernelModalProvider,
)


__all__ = [
    "GpuModeKernelFeedbackMutationProvider",
    "GpuModeKernelModalProvider",
    "MutationError",
]
