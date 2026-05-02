# `landscape_map/v2/`

Tool-calling LLM surrogate over GPU kernel speedup, with numerical
per-bin uncertainty.

The v2 surrogate produces a `KernelRuntimeEstimate` consisting of:
  - a `predicted_bin` (the most likely speedup bin among 1..8);
  - `bin_probabilities`, a **true probability simplex** over the eight
    success bins (renormalized at parse time);
  - `reasoning`, a short rationale string;
  - `raw_probability_sum`, the model's pre-renormalization sum, kept as
    a calibration-health signal.

This replaces v1's free-form text + 5-point Likert confidences with a
single tool call whose JSON arguments validate against a flat Pydantic
wire-format model. The same prompt structure as v1 — bin table,
ten-factor analysis guide, hardware-context table — is preserved
verbatim; only the "Confidence Scale" and "Output Format" sections of
the system prompt change.

## Backends

Three callers share the same prompt rendering, tool spec, and JSON
parser:

| Caller | Runs through | Module |
|---|---|---|
| Frontier API (Gemini) and Together gpt-oss baselines | LiteLLM `tools=` + forced `tool_choice` | `LlmSpeedupEstimator` |
| Trained-checkpoint scoring (calibration) | `tinker.SamplingClient` + cookbook `GptOssRenderer` | `TinkerSamplingClientEstimator` |
| RL `Env.step` during GRPO training | cookbook `Renderer.parse_response` directly | callers import `parse_tool_call_args` |

## Why a flat wire-format model?

Together's gpt-oss tool-call validator rejects JSON Schemas containing
`$defs`/`$ref`. The wire-format `SubmitEstimateArguments` therefore
flattens the eight per-bin probabilities into named `p_*` fields rather
than wrapping them in a sub-model. Inside the library we translate to
the more ergonomic `dict[SpeedupBin, float]` representation used by
`KernelRuntimeEstimate`.

## Why renormalize unconditionally?

The e0137 smoke run found that even with an explicit "must sum to 1"
instruction, models routinely return sums in `[0.75, 1.10]`. A
strict-tolerance parser drops ~3% of otherwise-usable predictions; an
unconditional-renormalize parser keeps every prediction with a positive
total mass and surfaces the original sum as a calibration-health
metric for downstream scoring.
