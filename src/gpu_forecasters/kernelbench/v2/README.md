# kernelbench / v2

Async-driver-shaped infrastructure for KernelBench search. Adapts the
KernelBench scoring/feedback contract to the per-candidate
`AsyncMutationProvider` / `AsyncEvaluationProvider` protocols consumed
by `gpu_forecasters.max_reward_puct.v2.search.SearchDriver`.

What lives here:

- `providers/modal_scoring.py` — `KernelBenchModalProvider`, async-chained
  over the split CPU-compile / GPU-benchmark Modal pipeline. One submit
  becomes one in-flight unit; the underlying coroutine awaits two
  `.remote.aio(...)` calls (compile, then bench) on a single asyncio
  loop running on a dedicated background thread.
- `providers/kernel_execution_feedback.py` —
  `KernelBenchFeedbackMutationProvider`, the per-candidate mutation
  provider. One `submit` issues exactly one outbound
  `litellm.acompletion(..., n=1)`, producing one code string or one
  `MutationError` for the v2 driver to log as `MutationFailed`.
- `providers/prompts/mutation.j2` — the mutation prompt template.
- `experiment_helper.py` — `run_l3_experiment` and the
  `ExperimentConfig` / `RunConfig` / `ProviderConfig` schemas that
  experiment cells (`experiments/eXXXX_kernelbench_l3_*`) consume.
- `l3_problems.py` — `L3ProblemReference` typed wrapper over
  `kernelbench.dataset.construct_kernelbench_dataset`, plus the
  Tier-B problem registry pinned by the gh070 paper testbed spec.

The v1 implementation under `gpu_forecasters/kernelbench/{modal_scoring,
modal_split_scoring, modal_image, scoring, core}.py` continues to back
the older `ModalProvider` and is what experiments e0015–e0021 still
import. Modules under `v2/` import from the v1 layer for unchanged
infrastructure (compute capability dict, Modal image, feedback type
hierarchy).
