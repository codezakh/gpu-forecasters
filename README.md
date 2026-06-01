# [GPU Forecasters: Language Models as Selective Surrogates for Kernel Runtime Optimization](https://arxiv.org/abs/2605.31464)

## Install

Clone with submodules and install in editable mode with [uv](https://github.com/astral-sh/uv):

```bash
git clone --recurse-submodules https://github.com/codezakh/gpu-surrogates.git
cd gpu-surrogates
uv sync
```

If you already cloned without `--recurse-submodules`, run `git submodule update --init --recursive` from the repo root.

Requires Python 3.13+ and a Linux machine with CUDA drivers (the lockfile pins a CUDA-128 PyTorch wheel). The package isn't published to PyPI — `uv sync` is the supported install path because two of the dependencies (`kernelbench`, `tinker-cookbook`) come from the vendored submodules under `third_party/`.

## `.env`

The runbook calls several external services. Put the keys you have in a `.env` file at the repo root:

```
MODAL_TOKEN_ID=...        # kernel measurement on Modal
MODAL_TOKEN_SECRET=...
TINKER_API_KEY=...        # GPT-OSS-20B sampling and GRPO training via Tinker
GEMINI_API_KEY=...        # Gemini-3 Flash / Pro
TOGETHER_API_KEY=...      # GPT-OSS-120B via Together
DEEPSEEK_API_KEY=...      # DeepSeek-V4
HF_TOKEN=...              # read the published artifacts on HuggingFace
```

You only need keys for the surrogates and stages you actually want to run. The per-script breakdown is in [`runbook/README.md`](runbook/README.md).

## Running the runbook

`runbook/` has seven numbered scripts that reproduce the paper's main results. Every script reads a JSON config and writes its outputs under `--output-dir`. Inputs come from HuggingFace (`codezakh/gpu-forecasters-*`); nothing needs to be downloaded by hand.

Verify the plumbing against the real backends with `--debug` (clamps to one row, one repeat, one search step):

```bash
uv run python runbook/01_score_baseline.py \
    --config runbook/configs/baseline_scoring/gemini3_flash.json \
    --output-dir runbook_output/01_smoke/ \
    --debug
```

Drop `--debug` for a full reproduction. Per-script docs, debug semantics, and how to chain `02 → 03` for a trained surrogate are in [`runbook/README.md`](runbook/README.md).

## Citation

```bibtex
@article{khan2026gpuforecasters,
  title={GPU Forecasters: Language Models as Selective Surrogates for Kernel Runtime Optimization},
  author={Khan, Zaid and Chen, Justin Chih-Yao and Cho, Jaemin and Stengel-Eskin, Elias and Bansal, Mohit},
  journal={arXiv preprint arXiv:2605.31464},
  year={2026}
}
```
