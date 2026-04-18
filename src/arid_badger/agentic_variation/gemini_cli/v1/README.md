# Gemini CLI agentic variation operator — v1

Programmatic harness around the Google Gemini CLI (`@google/gemini-cli`, pinned to `0.38.2`) used as an agentic variation operator against the TriMul kernel-optimization task.

The harness builds a minimal Node-based container with the CLI installed, starts a FastMCP scoring server on the host that publishes a single `score_trimul` tool backed by the Modal-based TriMul evaluator, and runs the CLI inside the container with `--network=host`. The agent iterates on candidate Triton kernels (`kernel_v1.py`, `kernel_v2.py`, ...), calling `score_trimul` each iteration. Every call is snapshotted server-side into `trajectory.jsonl`; the best-of-trajectory speedup is the run's result.

Two prompt surfaces — `system_prompt.md.j2` (role / methodology / tool guide) and `user_prompt.md.j2` (TriMul task instance) — are rendered per run. No `GEMINI.md`. Configuration for one run is captured in `ExperimentConfig` (model slug, GPU, Triton version, session-turn budget, aggregation method, optional thinking level) and persisted on the run's `TrimulRunResult` so each result carries the configuration that produced it.

Designed to be called by experiments that want to run the agent one or more times with a fixed budget and measure best-of-trajectory speedup against the TriMul benchmark.
