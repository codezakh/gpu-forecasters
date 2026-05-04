# balanced_pair_sampling / v1

Bin-balanced ordered-pair sampler for surrogate-training datasets.

## What this version does

Given one or more pools of correct candidate kernels (a `CandidateSource`
each), enumerate every ordered `(anchor, candidate)` pair within a
problem, label it with `log2(runtime_anchor / runtime_candidate)`,
bucket pairs by `SpeedupBin`, and emit at most `target_per_bin` pairs
per (problem, bin) cell, water-filled across problems so each
contributing problem gets its share before any one doubles up.

Each named source contributes the same `target_per_bin` per bin —
that's how the sampler enforces an equal split across sources. The
bin balance and source split are hard constraints; the only free knob
is `target_per_bin`, which determines dataset size: realized total ≈
`n_sources * 8 * target_per_bin`, modulo per-cell supply ceilings.

## When to make a v2

This version is appropriate to copy and modify if a future caller
needs to change one of the structural decisions, e.g.:

* a different binning scheme (the 8-bin `SpeedupBin` is hard-coded
  here)
* unequal source weighting (the v1 algorithm enforces equal `target_per_bin`
  across sources)
* alternate balance criterion (e.g. log-spaced ratio bands rather than
  bins, or per-problem caps rather than per-bin caps)

For changes that only affect the experiment that calls the sampler —
adding/removing a source, picking a different `target_per_bin`,
swapping the persistence format — modify the experiment, not this
module.
