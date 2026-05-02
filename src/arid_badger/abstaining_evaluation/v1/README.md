# `abstaining_evaluation/v1/`

Two surfaces for abstaining-surrogate evaluation share this module —
they share the `KernelRuntimeEstimate` and `KernelRuntimeQuery`
language but solve different problems.

## Surface 1: flat-eval metrics

Pure-data metrics for evaluating a surrogate as an abstaining classifier
over a held-out eval set. No LLM calls, no Modal, no I/O.

The surrogate is treated as a function from `KernelRuntimeQuery` to
`PredictOrAbstain`:

* if it predicts, we score the prediction via a pluggable `RiskFunction`;
* if it abstains, the user (would, in deployment) hit the real evaluator,
  so the abstained rows contribute their ground-truth runtime to a
  set-level regret risk and contribute zero to pointwise risks.

### Component-based design

* `ConfidenceScore` — `KernelRuntimeEstimate -> float`. Three concretes:
  `MaxProbScore`, `NegEntropyScore`, `Top2MarginScore`.
* `AbstainPolicy` — `KernelRuntimeEstimate | None -> PredictOrAbstain`.
  One concrete: `ThresholdAbstainPolicy(score, threshold)`. Parse
  failures (`estimate is None`) are forced abstentions regardless of
  threshold.
* `RiskFunction` — `(decisions, comparisons) -> float`. Three concretes:
  `BinaryMismatchRisk`, `SpeedupDistanceRisk`, `RegretRisk`.

### Risk-coverage curves

`risk_coverage_curve` sweeps a sequence of thresholds (the unique
confidence scores observed in the data, deduplicated and sorted) and
reports `(coverage, risk)` at each. `aurc` integrates that curve;
`selective_at_coverage` linearly interpolates.

`match_coverage_with_threshold` is the head-to-head helper for
comparing a native abstainer against a threshold abstainer at the same
realized coverage.

## Surface 2: search-embedded evaluation

Components for dropping an abstaining surrogate behind the v2 search
driver's `AsyncEvaluationProvider` seam. The surrogate either forecasts
the candidate's speedup (cheap LLM call, no GPU run) or defers to a
real evaluator (full Modal run). The search treats the two outcomes
uniformly through a discriminated union observation type and a single
reward channel.

### Domain types

* `CompoundObservation[T]` — the search-side observation. A
  discriminated union of:
  * `ForecastObservation`: `estimate: KernelRuntimeEstimate`,
    `expected_speedup: float`. No GPU run happened.
  * `RealObservation[T]`: `inner: GpuModeKernelObservation[T]`,
    `deferral_reason: str | None`. The real evaluator ran;
    `deferral_reason` is populated iff the surrogate triggered it.

### Reward seam

* `ForecastRewardPolicy` — protocol mapping a forecast to a real-eval-
  comparable reward.
* `ExpectedSpeedupReward` — the only concrete in v1.
  `Σ p_b · midpoint(b)` over the eight success bins, with bins 1 and 8
  using representative finite midpoints one half-octave past the
  closed edge.

### Providers

* `CompoundEvaluationProvider[T]` — implements
  `AsyncEvaluationProvider[CompoundObservation[T]]`. Composes a
  surrogate (`AbstainingLlmSpeedupEstimator`) and a real evaluator;
  forecasts resolve immediately, deferrals chain to the real
  evaluator's future via `add_done_callback`. Honors the V2 async
  intent — `submit` is non-blocking and many forecasts / deferrals
  can be in flight at once.
* `CompoundFeedbackMutationProvider[T]` — implements
  `AsyncMutationProvider[CompoundObservation[T]]`. Standalone (does
  not wrap the gpu_mode_kernel mutation provider): owns its own
  asyncio loop, semaphore, LiteLLM call, and prompt rendering.
  Dispatches on the parent observation arm to pick the right feedback
  prompt.

### Prompts

`prompts.py` is a sibling of `gpu_mode_kernel/prompts.py`, copied at
fork time so this module can evolve its mutation prompts without
coordinating. Renders three prompts:

* `build_base_prompt(pack, ...)` — kernel description + rules block.
* `format_real_eval_feedback_prompt(...)` — the four in-band
  `KernelExecutionFeedback` arms (compile / runtime / incorrect /
  success). Same shape as the gpu_mode_kernel formatter at fork time.
* `format_forecast_feedback_prompt(...)` — the surrogate-forecast
  arm. Same scaffolding (rules, base, parent code, closing rewrite
  instruction); the middle "evaluation result" block is replaced by
  a forecast block that explicitly marks itself as a prediction, not
  a measurement.
