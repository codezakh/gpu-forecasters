# `calibration/v2/`

Calibration metrics for surrogates that emit a true probability simplex
over the eight `landscape_map.v2` `SpeedupBin` success bins.

The v1 module assumes the surrogate emits a verbalized Likert label
per bin and projects to a numeric distribution via
`LikertNumericMapping` before scoring. v2 surrogates emit numerical
probabilities directly through the `submit_kernel_runtime_estimate`
tool call, so the projection step is a no-op and the metrics consume
`KernelRuntimeEstimate.bin_probabilities` directly.

What you get:

  - **scoring rules** — `brier`, `crps`, `nll` as pure functions over
    `dict[SpeedupBin, float]` and a true bin.
  - **ECE** — `reliability_bins_for` buckets parsed rows by
    `bin_probabilities[predicted_bin]` (the model's own claimed
    confidence in its argmax), and `expected_calibration_error` folds
    that into a scalar.
  - **Evaluator** — `evaluate_calibration(data)` returns a
    `CalibrationReport` aggregating accuracy, mean entropy, ECE, mean
    Brier / CRPS (parsed-only and full-set with a uniform-fallback
    penalty for parse failures), and the mean
    `raw_probability_sum` calibration-health signal.
