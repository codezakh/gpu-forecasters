# `abstaining_evaluation/v1/`

Pure-data metrics for evaluating a surrogate as an abstaining classifier
over a held-out eval set. No LLM calls, no Modal, no I/O.

The surrogate is treated as a function from `KernelRuntimeQuery` to
`PredictOrAbstain`:

* if it predicts, we score the prediction via a pluggable `RiskFunction`;
* if it abstains, the user (would, in deployment) hit the real evaluator,
  so the abstained rows contribute their ground-truth runtime to a
  set-level regret risk and contribute zero to pointwise risks.

## Component-based design

* `ConfidenceScore` — `KernelRuntimeEstimate -> float`. Three concretes:
  `MaxProbScore`, `NegEntropyScore`, `Top2MarginScore`.
* `AbstainPolicy` — `KernelRuntimeEstimate | None -> PredictOrAbstain`.
  One concrete: `ThresholdAbstainPolicy(score, threshold)`. Parse
  failures (`estimate is None`) are forced abstentions regardless of
  threshold.
* `RiskFunction` — `(decisions, comparisons) -> float`. Three concretes:
  `BinaryMismatchRisk`, `SpeedupDistanceRisk`, `RegretRisk`.

## Risk-coverage curves

`risk_coverage_curve` sweeps a sequence of thresholds (the unique
confidence scores observed in the data, deduplicated and sorted) and
reports `(coverage, risk)` at each. `aurc` integrates that curve;
`selective_at_coverage` linearly interpolates.

`match_coverage_with_threshold` is the head-to-head helper for
comparing a native abstainer against a threshold abstainer at the same
realized coverage.
