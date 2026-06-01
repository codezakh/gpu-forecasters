"""Run one §4.4 kernel-search comparison: standard or surrogate-filtered.

The §4.4 figure compares two PUCT search regimes at matched paid-eval
budget:

* **Standard**: every parent's ``samples_per_parent`` mutations are
  evaluated on GPU. ``k_per_parent`` is ignored at the selection
  barrier.
* **Surrogate-filtered**: ``samples_per_parent`` mutations are
  forecast by the surrogate; only the top ``k_per_parent`` (ranked by
  expected bin index) are evaluated on GPU. Paid evals match the
  standard mode at the budget knob, but the candidates promoted to
  paid eval are different.

``mode`` in the config selects which path runs. The surrogate is one
of the three published LoRA adapters or the bare base model.

Usage::

    python runbook/04_kernel_search.py \\
        --config runbook/configs/kernel_search/trimul__standard.json \\
        --output-dir runbook_output/04_trimul_standard/

    python runbook/04_kernel_search.py \\
        --config runbook/configs/kernel_search/trimul__surrogate_filtered.json \\
        --output-dir runbook_output/04_trimul_surrogate/ \\
        --debug
"""

from __future__ import annotations

import argparse
from pathlib import Path

from loguru import logger

from gpu_forecasters.experiment_utils import install_loguru_intercept
from gpu_forecasters.gpu_mode_kernel.experiment_helper import (
    ExperimentConfig,
    ProviderConfig,
    RunConfig,
    run_pack_experiment as run_pack_experiment_standard,
)
from gpu_forecasters.gpu_mode_kernel.surrogate_search.v1 import (
    A100_80GB_SXM4_HARDWARE,
    EvaluationConfig,
    MutatorConfig,
    SurrogateSearchExperimentConfig,
    TinkerSurrogateConfig,
    run_pack_experiment as run_pack_experiment_surrogate,
)
from gpu_forecasters.max_reward_puct.v2.config import SearchConfig as V2SearchConfig
from gpu_forecasters.max_reward_puct.v3.config import (
    ExpectedBinIndexRule,
    SearchConfig as V3SearchConfig,
)
from gpu_forecasters.runbook import (
    KernelSearchConfig,
    StandardSearchMode,
    SurrogateFilteredSearchMode,
    apply_debug_overrides,
)
from gpu_forecasters.runbook.packs import get_pack_runtime_and_case_type


def _run_standard(config: KernelSearchConfig, output_dir: Path) -> None:
    lib_config = ExperimentConfig(
        num_runs=1,
        run=RunConfig(
            search=V2SearchConfig(
                total_budget_steps=config.total_budget_steps,
                batch_size=config.batch_size,
                samples_per_parent=config.samples_per_parent,
                k_per_parent=config.k_per_parent,
                archive_capacity=config.archive_capacity,
                c_puct=config.c_puct,
                per_request_timeout_s=config.request_timeout_s,
            ),
            provider=ProviderConfig(
                model_slug=config.mutator_model_slug,
                gpu=config.hardware.gpu,
                aggregator=config.aggregator,
                max_llm_concurrency=config.max_llm_concurrency,
                max_tokens=None,
                request_timeout_s=config.request_timeout_s,
            ),
        ),
    )
    pack_runtime, case_speedup_type = get_pack_runtime_and_case_type(config.pack)
    run_pack_experiment_standard(
        pack_runtime=pack_runtime,
        case_speedup_type=case_speedup_type,
        output_dir=output_dir,
        config=lib_config,
    )


def _run_surrogate_filtered(
    config: KernelSearchConfig,
    mode: SurrogateFilteredSearchMode,
    output_dir: Path,
    checkpoint_uri: str | None,
) -> None:
    surrogate = mode.surrogate
    lib_config = SurrogateSearchExperimentConfig(
        search=V3SearchConfig(
            total_budget_steps=config.total_budget_steps,
            batch_size=config.batch_size,
            samples_per_parent=config.samples_per_parent,
            k_per_parent=config.k_per_parent,
            archive_capacity=config.archive_capacity,
            c_puct=config.c_puct,
            ranking_rule=ExpectedBinIndexRule(),
        ),
        mutator=MutatorConfig(
            model_slug=config.mutator_model_slug,
            max_llm_concurrency=config.max_llm_concurrency,
            request_timeout_s=config.request_timeout_s,
            max_tokens=None,
        ),
        surrogate=TinkerSurrogateConfig(
            base_model=surrogate.base_model,
            checkpoint_uri=checkpoint_uri,
            renderer_name=surrogate.renderer_name,
            temperature=surrogate.temperature,
            max_tokens=surrogate.max_tokens,
            max_retries=mode.max_retries,
        ),
        evaluation=EvaluationConfig(
            gpu=config.hardware.gpu,
            aggregator=config.aggregator,
            max_in_flight=config.max_llm_concurrency,
        ),
        hardware=A100_80GB_SXM4_HARDWARE,
        num_runs=1,
    )
    pack_runtime, case_speedup_type = get_pack_runtime_and_case_type(config.pack)
    run_pack_experiment_surrogate(
        pack_runtime=pack_runtime,
        case_speedup_type=case_speedup_type,
        output_dir=output_dir,
        config=lib_config,
    )


def _read_checkpoint_uri(artifact_path: Path) -> str:
    import json

    payload = json.loads(artifact_path.read_text())
    uri = payload.get("checkpoint_uri")
    if not isinstance(uri, str) or not uri.startswith("tinker://"):
        raise ValueError(
            f"{artifact_path} does not contain a Tinker checkpoint URI "
            f"(got checkpoint_uri={uri!r}). Run 02_train_surrogate.py "
            f"first to produce a training artifact."
        )
    return uri


def main() -> None:
    install_loguru_intercept()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a JSON config.")
    parser.add_argument("--output-dir", required=True, help="Where to write outputs.")
    parser.add_argument(
        "--surrogate-training-artifact",
        default=None,
        help=(
            "For surrogate-filtered mode: path to training_artifact.json "
            "from a 02_train_surrogate.py run. Omit to use the bare base "
            "model as the surrogate."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Plumbing-only smoke run: 1 step, 1 parent, 1 mutation.",
    )
    args = parser.parse_args()

    config = KernelSearchConfig.model_validate_json(Path(args.config).read_text())
    if args.debug:
        config = apply_debug_overrides(config)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(config.model_dump_json(indent=2))

    if isinstance(config.mode, StandardSearchMode):
        logger.info("mode=standard pack={} steps={}", config.pack, config.total_budget_steps)
        _run_standard(config, output_dir)
    elif isinstance(config.mode, SurrogateFilteredSearchMode):
        checkpoint_uri = (
            _read_checkpoint_uri(Path(args.surrogate_training_artifact))
            if args.surrogate_training_artifact
            else None
        )
        logger.info(
            "mode=surrogate_filtered pack={} steps={} checkpoint={}",
            config.pack,
            config.total_budget_steps,
            checkpoint_uri or "<base model>",
        )
        _run_surrogate_filtered(config, config.mode, output_dir, checkpoint_uri)
    else:  # pragma: no cover - exhaustive over the discriminated union
        raise TypeError(f"unknown search mode: {config.mode!r}")


if __name__ == "__main__":
    main()
