# `calibration/v1/`

Calibration metrics for an ordinal speedup-bin classifier whose output
is a verbalized Likert-confidence distribution over the eight
``SpeedupBin`` classes (failure bin excluded — surrogates here are
trained on success-only rows).

## What it computes

Given a held-out set of ``(true_bin, KernelRuntimeEstimate)`` triples:

* **CRPS** over the ordinal CDF, the proper scoring rule for ordered
  classes. Used both as a calibration metric *and*, in
  ``crps_calibration_reward``, as a reward term that an RL trainer can
  combine with the existing distance-on-``predicted_bin`` reward
  (RLCR-style).
* **Brier score** over the 8 classes treated as nominal. Reported
  alongside CRPS as a sanity-check; the two should move together but
  CRPS is the right primary metric.
* **ECE** (Expected Calibration Error) bucketed over
  ``bin_confidences[predicted_bin]``, the verbalized confidence in the
  argmax bin. This is the "if the model says high, is it right ~60% of
  the time?" question.
* **Reliability bins** for plotting reliability diagrams.
* **Sharpness** — mean entropy of the verbalized distribution. Without
  this, calibration improvements that come from spreading mass
  uniformly are indistinguishable from genuine improvements; reading
  ECE alongside sharpness disambiguates.
* **Accuracy** — argmax-bin accuracy. Reported alongside sharpness so
  drops in either are visible.

## Likert → numeric mapping

The verbalized Likert scale maps to a numeric prior at evaluation time
via ``LikertNumericMapping``. The mapping is a hyperparameter — small
changes shift Brier/CRPS values without changing rankings, so we log
the mapping inside every report.

## Failure modes

A ``KernelRuntimeEstimate`` of ``None`` means the rollout failed to
parse. Such rows contribute to ``parsed_rate`` and to scoring-rule
totals (CRPS / Brier scored against a uniform fallback distribution),
but are excluded from accuracy and ECE since ``predicted_bin`` is
undefined.
