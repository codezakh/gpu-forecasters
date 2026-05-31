# `surrogate_search/v1` — pack-bound entry point for v3 surrogate-filtered PUCT

This version exists so that a per-pack experiment file declares only
what is genuinely pack-specific — the `PackedModalRuntime` constant
and its `CaseSpeedupT` — and one `SurrogateSearchExperimentConfig`
that names the search budget, the mutator, the surrogate, and the
target device. The driver wiring (three providers, the event log, the
context-manager dance, the per-run summary computation, and the
multi-run skip-resume loop) lives behind one function call.

The surrogate is a discriminated union over `TinkerSurrogateConfig`
and `LiteLlmSurrogateConfig` because comparing a Tinker-served
trained checkpoint against a LiteLLM-served frontier model on the
same pack is the central comparison this harness supports. Encoding
the provider choice in the type lets the runner's surrogate
construction be exhaustive, and keeps provider-specific fields off
the variant that doesn't need them.

The pack and case-speedup type are runner arguments, not config
fields, because they are the subject of the experiment. A config
that specifies one but is run against a different pack would not be
meaningful, and conflating the two would invite sweep harnesses that
treat pack identity as just another hyperparameter.
