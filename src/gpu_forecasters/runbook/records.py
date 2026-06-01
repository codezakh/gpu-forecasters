"""Pydantic record types matching the on-HuggingFace dataset schemas.

The runbook loaders parse raw HF rows into these typed objects. Each
field carries the same name and meaning as on the published dataset
card. Pack-internal hashes (``comparison_id`` etc.) and the dual
provenance fields (``source_search`` + ``internal_experiment``) are
preserved through scoring outputs so a reader can join a forecast
back to the row that produced it.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


_FROZEN = ConfigDict(frozen=True)


class BinProbability(BaseModel):
    model_config = _FROZEN

    bin: int = Field(ge=1, le=8)
    p: float = Field(ge=0.0, le=1.0)


class EvalSetRow(BaseModel):
    """One row of ``codezakh/gpu-forecasters-eval-set``."""

    model_config = _FROZEN

    comparison_id: str
    pack: str
    anchor_code: str
    candidate_code: str
    hardware: str
    aggregated_speedup: float
    true_bin: int = Field(ge=1, le=8)
    source_id: str
    source_search: str
    internal_experiment: str


class ScoredEvalRow(BaseModel):
    """One row of ``codezakh/gpu-forecasters-eval-set-predictions``."""

    model_config = _FROZEN

    comparison_id: str
    pack: str
    surrogate_label: str
    repeat: int = Field(ge=0)
    predicted_bin: int | None = None
    bin_probabilities: tuple[BinProbability, ...]
    reasoning: str
    raw_probability_sum: float | None
    parse_failed: bool
    parse_error: str | None
    input_tokens: int | None
    output_tokens: int | None
    elapsed_s: float


class RlTrainingRow(BaseModel):
    """One row of ``codezakh/gpu-forecasters-rl-training-pool``."""

    model_config = _FROZEN

    row_id: str
    pack: str
    anchor_code: str
    candidate_code: str
    aggregated_speedup: float
    relative_bin: int = Field(ge=1, le=8)
    pair_type: Literal["seed", "parent_edit", "pair"]
    hardware: str
    source_id: str
    source_search: str
    internal_experiment: str


class DiscoveryPairRow(BaseModel):
    """One row of ``codezakh/gpu-forecasters-discovery-pairs``."""

    model_config = _FROZEN

    pair_id: str
    problem_id: str
    benchmark_family: Literal["gpu_mode", "kernelbench_l3"]
    parent_code: str
    child_code: str
    g_speedup: float
    true_bin: int = Field(ge=1, le=8)
    hardware: str
    parent_node_id: str
    child_node_id: str
    source_id: str
    source_search: str
    internal_experiment: str


__all__ = [
    "BinProbability",
    "DiscoveryPairRow",
    "EvalSetRow",
    "RlTrainingRow",
    "ScoredEvalRow",
]
