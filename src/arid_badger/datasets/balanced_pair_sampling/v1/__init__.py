"""Bin-balanced ordered-pair sampler for surrogate training datasets.

Public surface:

  - :class:`CandidateKernel`, :class:`LabeledPair`, :data:`ProblemId` —
    domain types.
  - :class:`CandidateSource` protocol, plus concretes
    :class:`SakanaCandidateSource` and
    :class:`GpuModeSeedAnchoredCandidateSource`.
  - :class:`BalancedPairSamplerConfig`, :class:`BalancedPairDataset`,
    :class:`SidedSampleReport`, :class:`PerProblemSupply` — config and
    result types.
  - :func:`build_balanced_pair_dataset` — the entry point.

Why this lives in the library and not in one experiment: every
training experiment that wants a different dataset size calls this
with a different ``target_per_bin``. The sampler's behavior is
versioned (``v1/``) so a change in algorithm cannot silently invalidate
historical runs — a v2 would live alongside v1.
"""

from .domain import CandidateKernel, LabeledPair, ProblemId
from .sampler import (
    BalancedPairDataset,
    BalancedPairSamplerConfig,
    PerProblemSupply,
    SidedSampleReport,
    build_balanced_pair_dataset,
)
from .sources import (
    CandidateSource,
    GpuModeSeedAnchoredCandidateSource,
    SakanaCandidateSource,
)


__all__ = [
    "BalancedPairDataset",
    "BalancedPairSamplerConfig",
    "CandidateKernel",
    "CandidateSource",
    "GpuModeSeedAnchoredCandidateSource",
    "LabeledPair",
    "PerProblemSupply",
    "ProblemId",
    "SakanaCandidateSource",
    "SidedSampleReport",
    "build_balanced_pair_dataset",
]
