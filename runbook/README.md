# Runbook

Seven scripts that reproduce the paper's main results from a fresh
checkout. Each one reads a JSON config and calls into the library
(`gpu_forecasters.*`) to do the work; pass `--debug` to run a small
version first. The datasets and trained adapters live on HuggingFace.

## Quickstart

Install per the top-level [README](../README.md), fill in `.env` with the keys listed under [Environment](#environment) below, then:

```bash
# Quick check that it runs end-to-end against the real services:
uv run --env-file .env python runbook/01_score_baseline.py \
    --config runbook/configs/baseline_scoring/gemini3_flash.json \
    --output-dir runbook_output/01_smoke/ \
    --debug

# Reproduce one full result from the paper (≈ 4 hours on the held-out eval set):
uv run --env-file .env python runbook/01_score_baseline.py \
    --config runbook/configs/baseline_scoring/gemini3_flash.json \
    --output-dir runbook_output/01_gemini3_flash/
```

## Training your own surrogate and scoring it

Tinker checkpoint URIs are private to your account. The adapters we
published on HuggingFace are for serving on vLLM or SGLang — you can't
load them back into Tinker. So to score a trained surrogate, first
train one in your own Tinker account (`02`), then score it (`03`):

```bash
# Train a correctness-reward surrogate. Writes training_artifact.json,
# which holds a tinker://... URI.
uv run --env-file .env python runbook/02_train_surrogate.py \
    --config runbook/configs/training/correctness.json \
    --output-dir runbook_output/02_correctness/

# Score it. --training-artifact is the file the training run above wrote.
uv run --env-file .env python runbook/03_score_trained.py \
    --config runbook/configs/trained_scoring/correctness.json \
    --training-artifact runbook_output/02_correctness/training_artifact.json \
    --output-dir runbook_output/03_correctness/
```

The kernel search that filters candidates with a surrogate works the same way:

```bash
uv run --env-file .env python runbook/04_kernel_search.py \
    --config runbook/configs/kernel_search/trimul__surrogate_filtered.json \
    --surrogate-training-artifact runbook_output/02_correctness/training_artifact.json \
    --output-dir runbook_output/04_trimul_surrogate/
```

## Scripts

| Script | What it reproduces | Default cost |
| --- | --- | --- |
| `00_upstream_puct.py` | One PUCT search of the kind that generated the raw data. Included to show where the data came from — most people will just download it from HuggingFace. | ~30 GPU-hours per pack |
| `01_score_baseline.py` | An off-the-shelf model scored on the held-out eval set. | LLM calls over the eval set, ×3 repeats |
| `02_train_surrogate.py` | One GRPO-trained surrogate (correctness, +Brier, or +CRPS). Writes a Tinker checkpoint URI to `training_artifact.json`. | ~6 Tinker-hours |
| `03_score_trained.py` | A trained surrogate scored on the held-out eval set. Reads its checkpoint URI from a `02` run via `--training-artifact`. | Same as 01 |
| `04_kernel_search.py` | One budget-matched kernel search, standard or surrogate-filtered. Surrogate-filtered mode reads the checkpoint URI via `--surrogate-training-artifact`. | Same as 00 |
| `05_score_discovery.py` | A surrogate scored on the discovery-pair eval set. | Smaller than 01 |
| `06_make_figures.py` | Redraws the paper figures from the HuggingFace data. No GPU or LLM needed. | Seconds |

## `--debug` mode

Every script accepts `--debug`, which runs the smallest job that still
calls the real services end-to-end. Nothing is mocked — it's there to
confirm your setup works before you pay for a full run.

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

JSON, parsed by Pydantic. Each field is documented on its model in
[`gpu_forecasters.runbook.configs`](../src/gpu_forecasters/runbook/configs.py).

```
configs/
├── upstream_puct/
│   └── gpu_mode/                       # one config per source PUCT search
├── baseline_scoring/                   # off-the-shelf models
├── training/                           # GRPO reward variants
├── trained_scoring/                    # trained LoRA adapters
├── kernel_search/                      # standard vs surrogate-filtered search
├── discovery_scoring/                  # scoring on the discovery pairs
└── figures/                            # which figures to draw, and from which surrogates
```

Two configs that differ in any field reproduce two different results.
The file name spells out what's different (`correctness_brier.json`,
`trimul__surrogate_filtered.json`) so you can tell runs apart without
opening them.

## HuggingFace artifacts

All datasets and models live under `codezakh/` on huggingface.co;
[`data_pointers.json`](data_pointers.json) lists them.

Datasets:

* `codezakh/gpu-forecasters-eval-set` — held-out eval pairs
* `codezakh/gpu-forecasters-eval-set-predictions` — pre-computed forecasts
* `codezakh/gpu-forecasters-rl-training-pool` — GRPO training rows
* `codezakh/gpu-forecasters-discovery-pairs` — discovery pairs (parent and child kernels)
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

Every row from a PUCT search carries two fields:

* `source_search` — the config path that reproduces it (e.g.
  `gpu_mode/cross_entropy__e0091__gemini3_flash.json`).
* `internal_experiment` — the experiment name from our private repo.
  It's there for our own tracing; you can ignore it.
