# `eval_dataset_builder/v1/`

Tooling to build held-out kernel evaluation datasets bin-by-bin. Given a
``KernelPack`` and a source of harvested kernels (from prior searches),
this module fills under-populated speedup bins by running a
goal-conditioned PUCT search per bin, then writes a JSONL dataset plus a
provenance manifest.

The eval dataset's purpose: hold out a stratified pool of correct
kernels — covering the full range of speedup outcomes — that downstream
tools (speedup surrogates, ablations, predictive evaluators) can score
against on stable ground truth, independent of any single search run's
trajectory.

The module is pack-generic over ``KernelPack[TestArgsT, CaseSpeedupT]``.
The harvest stage is decoupled via the ``HarvestedKernelSource`` protocol
so the library does not depend on any specific source-search checkpoint
format — callers ship an adapter that yields ``KernelRuntimeComparison``
rows from whatever shape their prior search produced.
