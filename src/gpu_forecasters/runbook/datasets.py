"""HuggingFace artifact URIs and loaders for runbook scripts.

The runbook never reads from local disk — it talks to HuggingFace by
repo ID so a reader who clones the public repo can reproduce a paper
result from a clean machine.

Functions here return Pydantic-typed Python objects rather than raw
``datasets.Dataset`` rows. The on-disk schema is mirrored by classes
in :mod:`gpu_forecasters.runbook.records`.
"""

from __future__ import annotations

from typing import Iterable, Literal

from gpu_forecasters.eval_dataset_builder.v1 import KernelRuntimeComparison
from gpu_forecasters.landscape_map.v1.domain import (
    HardwareContext as HardwareContextV1,
    SpeedupBin,
)
from gpu_forecasters.runbook.records import (
    DiscoveryPairRow,
    EvalSetRow,
    RlTrainingRow,
)


# ---------------------------------------------------------------------------
# Canonical HF artifact identifiers
# ---------------------------------------------------------------------------


HF_EVAL_SET = "codezakh/gpu-forecasters-eval-set"
HF_EVAL_SET_PREDICTIONS = "codezakh/gpu-forecasters-eval-set-predictions"
HF_RL_TRAINING_POOL = "codezakh/gpu-forecasters-rl-training-pool"
HF_DISCOVERY_PAIRS = "codezakh/gpu-forecasters-discovery-pairs"
HF_PUCT_SEARCH_EVENTS = "codezakh/gpu-forecasters-puct-search-events"

HF_LORA_REPOS: dict[str, str] = {
    "correctness": "codezakh/gpu-forecasters-gpt-oss-20b-correctness",
    "correctness_brier": "codezakh/gpu-forecasters-gpt-oss-20b-correctness-brier",
    "correctness_crps": "codezakh/gpu-forecasters-gpt-oss-20b-correctness-crps",
}


_DEFAULT_HARDWARE_DEVICE = "NVIDIA A100-SXM4-80GB"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _build_hardware(device_name: str) -> HardwareContextV1:
    """Reconstruct a v1 ``HardwareContext`` for one of the paper devices.

    The HF eval-set rows carry only ``hardware`` as a string device
    name; the surrogate-scoring estimators need the full struct. The
    paper uses one device per pack and that device is documented in the
    dataset card, so this lookup is short and explicit.
    """
    if device_name == "NVIDIA A100-SXM4-80GB":
        return HardwareContextV1(
            device_name=device_name,
            compute_capability=(8, 0),
            total_global_memory_gb=80.0,
            multiprocessor_count=108,
            max_threads_per_multiprocessor=2048,
            clock_rate_ghz=1.41,
            memory_clock_rate_ghz=1.512,
            memory_bus_width_bits=5120,
        )
    if device_name == "NVIDIA L40S":
        return HardwareContextV1(
            device_name=device_name,
            compute_capability=(8, 9),
            total_global_memory_gb=44.5,
            multiprocessor_count=142,
            max_threads_per_multiprocessor=1536,
            clock_rate_ghz=2.52,
            memory_clock_rate_ghz=9.0,
            memory_bus_width_bits=384,
        )
    raise ValueError(
        f"unknown hardware device_name: {device_name!r}. Known devices: "
        f"A100-SXM4-80GB, L40S. Add a branch to runbook.datasets if a new "
        f"device joins the paper."
    )


def _eval_row_to_comparison(row: EvalSetRow) -> KernelRuntimeComparison:
    return KernelRuntimeComparison(
        source_id=row.source_id,
        reference_code=row.anchor_code,
        candidate_code=row.candidate_code,
        hardware=_build_hardware(row.hardware),
        aggregated_speedup=row.aggregated_speedup,
        true_bin=SpeedupBin(row.true_bin),
    )


def _pair_row_to_comparison(row: DiscoveryPairRow) -> KernelRuntimeComparison:
    """A discovery row's anchor is the parent and candidate is the child."""
    return KernelRuntimeComparison(
        source_id=row.pair_id,
        reference_code=row.parent_code,
        candidate_code=row.child_code,
        hardware=_build_hardware(row.hardware),
        aggregated_speedup=row.g_speedup,
        true_bin=SpeedupBin(row.true_bin),
    )


def load_canonical_eval_set(
    *,
    packs: Iterable[str] | None = None,
    revision: str = "main",
) -> dict[str, list[tuple[EvalSetRow, KernelRuntimeComparison]]]:
    """Load the canonical eval set from HF, grouped by pack name.

    Returns a dict keyed by pack name. Each value is a list of
    ``(row, comparison)`` pairs — the raw HF row (kept for provenance
    fields and the content-addressable ``comparison_id``) alongside the
    library-typed ``KernelRuntimeComparison`` the estimator expects.

    ``packs=None`` loads every published pack via the ``combined``
    config, then partitions client-side. Passing an explicit ``packs``
    iterable loads one HF config per pack instead.
    """
    from datasets import load_dataset

    grouped: dict[str, list[tuple[EvalSetRow, KernelRuntimeComparison]]] = {}
    if packs is None:
        ds = load_dataset(HF_EVAL_SET, name="combined", split="eval", revision=revision)
        for raw in ds:
            row = EvalSetRow.model_validate(raw)
            grouped.setdefault(row.pack, []).append((row, _eval_row_to_comparison(row)))
        return grouped

    for pack in packs:
        ds = load_dataset(HF_EVAL_SET, name=pack, split="eval", revision=revision)
        rows: list[tuple[EvalSetRow, KernelRuntimeComparison]] = []
        for raw in ds:
            row = EvalSetRow.model_validate(raw)
            rows.append((row, _eval_row_to_comparison(row)))
        grouped[pack] = rows
    return grouped


def load_discovery_pairs(
    *,
    families: Iterable[Literal["gpu_mode", "kernelbench_l3"]] | None = None,
    revision: str = "main",
) -> dict[str, list[tuple[DiscoveryPairRow, KernelRuntimeComparison]]]:
    """Load discovery pairs from HF, grouped by ``problem_id``.

    Each row carries an explicit ``problem_id`` (pack name for
    GPU Mode, KernelBench problem slug for L3). The scoring runner
    groups by that so the per-pack output layout used by §4.5 stays
    intact.
    """
    from datasets import load_dataset

    selected = tuple(families) if families is not None else ("gpu_mode", "kernelbench_l3")
    grouped: dict[str, list[tuple[DiscoveryPairRow, KernelRuntimeComparison]]] = {}
    for family in selected:
        ds = load_dataset(HF_DISCOVERY_PAIRS, name=family, split="pairs", revision=revision)
        for raw in ds:
            row = DiscoveryPairRow.model_validate(raw)
            grouped.setdefault(row.problem_id, []).append(
                (row, _pair_row_to_comparison(row))
            )
    return grouped


def load_rl_training_pool(
    *,
    packs: Iterable[str] | None = None,
    revision: str = "main",
) -> list[RlTrainingRow]:
    """Load the RL training pool from HF as a flat list of typed rows."""
    from datasets import load_dataset

    rows: list[RlTrainingRow] = []
    if packs is None:
        ds = load_dataset(HF_RL_TRAINING_POOL, name="combined", split="train", revision=revision)
        for raw in ds:
            rows.append(RlTrainingRow.model_validate(raw))
        return rows

    for pack in packs:
        ds = load_dataset(HF_RL_TRAINING_POOL, name=pack, split="train", revision=revision)
        for raw in ds:
            rows.append(RlTrainingRow.model_validate(raw))
    return rows


__all__ = [
    "HF_DISCOVERY_PAIRS",
    "HF_EVAL_SET",
    "HF_EVAL_SET_PREDICTIONS",
    "HF_LORA_REPOS",
    "HF_PUCT_SEARCH_EVENTS",
    "HF_RL_TRAINING_POOL",
    "load_canonical_eval_set",
    "load_discovery_pairs",
    "load_rl_training_pool",
]
