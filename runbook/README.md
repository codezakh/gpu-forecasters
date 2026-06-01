# Runbook

Seven scripts that reproduce the paper's main results from a clean
checkout. Each script reads a JSON config, optionally clamps to a
debug-sized subset, and dispatches to existing library code.

The runbook is a thin orchestration layer. The library
(`gpu_forecasters.*`) owns the actual search loops, estimators, and
training code. Data and trained adapters live on HuggingFace.

## Quickstart

Install per the top-level [README](../README.md), populate `.env` with the keys listed in [Environment](#environment) below, then:

```bash
# Confirm plumbing works against the real backends:
uv run --env-file .env python runbook/01_score_baseline.py \
    --config runbook/configs/baseline_scoring/gemini3_flash.json \
    --output-dir runbook_output/01_smoke/ \
    --debug

# Reproduce a full paper cell (this one ≈ 4 hours on the canonical eval set):
uv run --env-file .env python runbook/01_score_baseline.py \
    --config runbook/configs/baseline_scoring/gemini3_flash.json \
    --output-dir runbook_output/01_gemini3_flash/
```

## Training your own surrogate and scoring it

Tinker checkpoint URIs are account-private. The published HF LoRA
adapters are archival artifacts for external (vLLM / SGLang) serving;
they cannot be loaded back into Tinker. To run our pipeline with a
trained surrogate, train one in your own Tinker account first and
chain `02 → 03`:

```bash
# Train a correctness-reward surrogate. Writes training_artifact.json
# with a tinker://... URI inside it.
uv run --env-file .env python runbook/02_train_surrogate.py \
    --config runbook/configs/training/correctness.json \
    --output-dir runbook_output/02_correctness/

# Score it. --training-artifact is the URI carrier.
uv run --env-file .env python runbook/03_score_trained.py \
    --config runbook/configs/trained_scoring/correctness.json \
    --training-artifact runbook_output/02_correctness/training_artifact.json \
    --output-dir runbook_output/03_correctness/
```

Same pattern for the §4.4 surrogate-filtered kernel search:

```bash
uv run --env-file .env python runbook/04_kernel_search.py \
    --config runbook/configs/kernel_search/trimul__surrogate_filtered.json \
    --surrogate-training-artifact runbook_output/02_correctness/training_artifact.json \
    --output-dir runbook_output/04_trimul_surrogate/
```

## Scripts

| Script | What it reproduces | Default cost |
| --- | --- | --- |
| `00_upstream_puct.py` | One upstream PUCT search that produced the raw data archive. Provenance only — most readers will pull the HF artifacts. | ~30 GPU-hours per pack |
| `01_score_baseline.py` | One off-the-shelf surrogate scored on the canonical eval set (§4.3, Tab. 1, Figs 1–3) | ~LLM calls × 424 × 3 |
| `02_train_surrogate.py` | One GRPO-trained surrogate variant (correctness / +Brier / +CRPS) — produces a Tinker checkpoint URI in `training_artifact.json` | ~6 Tinker-hours |
| `03_score_trained.py` | A trained surrogate scored on the canonical eval set (§4.3). Reads the checkpoint URI from an upstream `02` run's `training_artifact.json` via `--training-artifact`. | Same as 01 |
| `04_kernel_search.py` | One §4.4 budget-matched kernel search (standard vs surrogate-filtered). Surrogate-filtered mode reads the checkpoint URI via `--surrogate-training-artifact`. | Same as 00 |
| `05_score_discovery.py` | One surrogate scored on the discovery-pair eval set (§4.5, Fig. 5) | Smaller than 01 |
| `06_make_figures.py` | Regenerates the paper figures from the published HF artifacts; no GPU or LLM required | Seconds |

## `--debug` mode

Every script accepts `--debug`. The flag scopes the workload to the
smallest configuration that still exercises the real execution path.
Backends are *not* stubbed — the goal is plumbing verification, not
unit-testing.

| Script | What `--debug` does |
| --- | --- |
| `00_upstream_puct.py` | `total_budget_steps=1`, `batch_size=1`, `samples_per_parent=1`, `k_per_parent=1`, `max_llm_concurrency=1` |
| `01_score_baseline.py` | 1 row per pack, 1 repeat, `max_concurrency=1` |
| `02_train_surrogate.py` | `num_iters=1`, `group_size=2`, `groups_per_batch=1`, `max_tokens=1024`, 4 training rows |
| `03_score_trained.py` | 1 row per pack, 1 repeat |
| `04_kernel_search.py` | Same as 00 |
| `05_score_discovery.py` | 5 pairs, 1 repeat, `max_concurrency=1` |
| `06_make_figures.py` | Reads `runbook/fixtures/scored_eval_tiny.jsonl` instead of pulling HF |

## Config layout

JSON, parsed by Pydantic. Field descriptions live in
[`gpu_forecasters.runbook.configs`](../src/gpu_forecasters/runbook/configs.py)
on the corresponding model.

```
configs/
├── upstream_puct/
│   └── gpu_mode/                       # one config per source PUCT search
├── baseline_scoring/                   # off-the-shelf surrogates (§4.3)
├── training/                           # GRPO reward variants (§4.3)
├── trained_scoring/                    # trained LoRA adapters (§4.3)
├── kernel_search/                      # standard vs surrogate-filtered PUCT (§4.4)
├── discovery_scoring/                  # discovery-pair scoring (§4.5)
└── figures/                            # which figures to render and from which surrogates
```

Two configs that differ in any field are two different reproductions.
File names carry the discriminator (`correctness_brier.json`,
`trimul__surrogate_filtered.json`) so a reader can tell which run a
file produces without opening it.

## HuggingFace artifacts

All datasets and models live under `codezakh/` on huggingface.co.
The canonical pointers are in [`data_pointers.json`](data_pointers.json).
The published artifacts are pinned at `revision="main"` until a paper
release tags them `v1.0`.

Datasets:

* `codezakh/gpu-forecasters-eval-set` — 424 held-out eval pairs
* `codezakh/gpu-forecasters-eval-set-predictions` — pre-computed forecasts
* `codezakh/gpu-forecasters-rl-training-pool` — GRPO training rows
* `codezakh/gpu-forecasters-discovery-pairs` — §4.5 parent-child pairs
* `codezakh/gpu-forecasters-puct-search-events` — raw search events

Models (LoRA adapters for `openai/gpt-oss-20b`):

* `codezakh/gpu-forecasters-gpt-oss-20b-correctness`
* `codezakh/gpu-forecasters-gpt-oss-20b-correctness-brier`
* `codezakh/gpu-forecasters-gpt-oss-20b-correctness-crps`

## Environment

| Variable | Used by | Affected scripts |
| --- | --- | --- |
| `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET` | kernel measurement | 00, 04 |
| `TINKER_API_KEY` | GPT-OSS-20B sampling and training | 01 (oss-20b row), 02, 03, 04 (surrogate-filtered) |
| `GEMINI_API_KEY` | Gemini-3 Flash / Pro | 00, 01, 04, 05 |
| `TOGETHER_API_KEY` | GPT-OSS-120B | 01, 05 |
| `DEEPSEEK_API_KEY` | DeepSeek-V4 | 01, 05 |
| `HF_TOKEN` | HF dataset / model reads | every script that pulls HF |

## Provenance

Every row that came out of a PUCT trace carries two fields:

* `source_search` — the runbook config path that would reproduce the
  trace (e.g. `gpu_mode/cross_entropy__e0091__gemini3_flash.json`).
* `internal_experiment` — the original private-repo experiment slug,
  kept for audit. Public readers can ignore it.
