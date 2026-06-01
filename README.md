# [GPU Forecasters: Language Models as Selective Surrogates for Kernel Runtime Optimization](https://arxiv.org/abs/2605.31464)

## Install

Clone with submodules and install in editable mode with [uv](https://github.com/astral-sh/uv):

```bash
git clone --recurse-submodules https://github.com/codezakh/gpu-surrogates.git
cd gpu-surrogates
uv sync
```

If you already cloned without `--recurse-submodules`, run `git submodule update --init --recursive` from the repo root.

Requires Python 3.13+ and a Linux machine with CUDA drivers.

## Verifying install

```bash
uv run pytest
```

Tests that need Modal or a CUDA GPU are excluded by default. This command does not need any API keys to run.

## API keys

The runbook scripts call Modal, Tinker, and various LLM providers. Put the keys you have in a `.env` file at the repo root:

```
MODAL_TOKEN_ID=...        # kernel measurement on Modal
MODAL_TOKEN_SECRET=...
TINKER_API_KEY=...        # GPT-OSS-20B sampling and GRPO training via Tinker
GEMINI_API_KEY=...        # Gemini-3 Flash / Pro
TOGETHER_API_KEY=...      # GPT-OSS-120B via Together
DEEPSEEK_API_KEY=...      # DeepSeek-V4
HF_TOKEN=...              # read the published artifacts on HuggingFace
```

You only need keys for services you actually use. The per-script breakdown is in [`runbook/README.md`](runbook/README.md).

## Running the runbook

`runbook/` has seven numbered scripts that reproduce the paper's main results. Every script reads a JSON config and writes its outputs under `--output-dir`. Inputs come from HuggingFace. Nothing needs to be downloaded by hand.

Every script accepts `--debug` for a fast test run:

```bash
uv run python runbook/01_score_baseline.py \
    --config runbook/configs/baseline_scoring/gemini3_flash.json \
    --output-dir runbook_output/01_smoke/ \
    --debug
```

Remove `--debug` for the full run. See [`runbook/README.md`](runbook/README.md) for details on each script.

## Resuming after a crash

Long-running scripts save their progress as they go and skip work that's already done when you rerun them. Use the same `--config` and `--output-dir` to continue from where you stopped. There is no resume flag.

- PUCT searches (`00`, `04`): the script writes an event log at `events.jsonl`. When you rerun it, the script replays the log and continues mid-step.
- Surrogate scoring (`01`, `03`, `05`): each (surrogate, repeat, row) writes its own cache file. When you rerun, already-scored rows are skipped.
- Training (`02`): writes a `training_artifact.json` with the Tinker checkpoint URI. The checkpoint itself lives on Tinker's servers.

## HuggingFace artifacts

Datasets:

| Repo | Contents |
| --- | --- |
| [`codezakh/gpu-forecasters-eval-set`](https://huggingface.co/datasets/codezakh/gpu-forecasters-eval-set) | 424 held-out (reference, candidate) kernel pairs |
| [`codezakh/gpu-forecasters-eval-set-predictions`](https://huggingface.co/datasets/codezakh/gpu-forecasters-eval-set-predictions) | Surrogate forecasts on the eval set |
| [`codezakh/gpu-forecasters-rl-training-pool`](https://huggingface.co/datasets/codezakh/gpu-forecasters-rl-training-pool) | GRPO training rows |
| [`codezakh/gpu-forecasters-discovery-pairs`](https://huggingface.co/datasets/codezakh/gpu-forecasters-discovery-pairs) | Parent-child pairs for the §4.5 discovery evaluation |
| [`codezakh/gpu-forecasters-puct-search-events`](https://huggingface.co/datasets/codezakh/gpu-forecasters-puct-search-events) | Raw event logs from the kernel searches |

LoRA adapters for `openai/gpt-oss-20b`:

| Repo | Reward |
| --- | --- |
| [`codezakh/gpu-forecasters-gpt-oss-20b-correctness`](https://huggingface.co/codezakh/gpu-forecasters-gpt-oss-20b-correctness) | correctness |
| [`codezakh/gpu-forecasters-gpt-oss-20b-correctness-brier`](https://huggingface.co/codezakh/gpu-forecasters-gpt-oss-20b-correctness-brier) | correctness + Brier |
| [`codezakh/gpu-forecasters-gpt-oss-20b-correctness-crps`](https://huggingface.co/codezakh/gpu-forecasters-gpt-oss-20b-correctness-crps) | correctness + CRPS |

## Where things live

Surrogate and search:

- `src/gpu_forecasters/landscape_map/v1/`: the surrogate prompt template, the response parser, and the code that classifies predicted speedups into one of 8 categories.
- `src/gpu_forecasters/max_reward_puct/v3/`: the PUCT search code. Picks the next kernel to mutate, runs it, keeps the best in an archive.
- `src/gpu_forecasters/gpu_mode_kernel/surrogate_search/v1/`: surrogate-filtered PUCT. Each candidate is first scored by the surrogate. Only candidates the surrogate predicts will be faster are run on a real GPU.

Kernel evaluation on Modal:

- `src/gpu_forecasters/gpu_mode_kernel/`: code that runs candidate kernels on the six GPU Mode tasks (TriMul, cross-entropy, three Gated DeltaNet variants, FP8 quantization). One file per task in `packs/`. Shared evaluation code in `modal_scoring.py`.
- `src/gpu_forecasters/kernelbench/`: runs the full-network problems from KernelBench (VGG19, SwinMLP, EfficientNet, and other PyTorch models).

Training:

- `src/gpu_forecasters/runbook/training.py`: the GRPO trainer and the three reward functions (correctness, +Brier, +CRPS).

Data and runbook library:

- `src/gpu_forecasters/eval_dataset_builder/v1/`: builds the held-out evaluation set published as `gpu-forecasters-eval-set` from the PUCT search event logs.
- `src/gpu_forecasters/runbook/`: the shared library the runbook scripts use. Pydantic config types, HuggingFace dataset loaders, and the `--debug` override logic.

## Reporting issues

[github.com/codezakh/gpu-surrogates/issues](https://github.com/codezakh/gpu-surrogates/issues)

## Acknowledgements

This work uses and takes inspiration from [KernelBench](https://github.com/ScalingIntelligence/KernelBench) and [ttt-discover](https://github.com/test-time-training/discover). Thanks to [Modal](https://modal.com) for an academic compute grant.

## Citation

```bibtex
@article{khan2026gpuforecasters,
  title={GPU Forecasters: Language Models as Selective Surrogates for Kernel Runtime Optimization},
  author={Khan, Zaid and Chen, Justin Chih-Yao and Cho, Jaemin and Stengel-Eskin, Elias and Bansal, Mohit},
  journal={arXiv preprint arXiv:2605.31464},
  year={2026}
}
```
