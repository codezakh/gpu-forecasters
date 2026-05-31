"""Durable-log-backed results loader for v2.

Reads only ``rollouts.jsonl``. No PUCT snapshot parsing, no
``metrics.jsonl`` dependency — everything downstream analysis needs is
already on ``RolloutRecord``. The loader's folds (best-by-step,
success filter, summary) are the canonical shape experiment-level
plots consume.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from gpu_forecasters.trimul.core import SuccessFeedback
from gpu_forecasters.ttt_discover.v2.domain.records import RolloutRecord


class BestByStep(BaseModel):
    model_config = ConfigDict(frozen=True)

    step: int
    best_reward: float
    best_candidate_id: str


class SeedSummary(BaseModel):
    model_config = ConfigDict(frozen=True)

    seed_index: int
    num_rollouts: int
    num_successes: int
    num_parse_failures: int
    num_compile_failures: int
    num_runtime_errors: int
    num_incorrect: int
    num_infra_failures: int
    best_reward: float | None


def _parse_rollouts(path: Path) -> list[RolloutRecord]:
    records: list[RolloutRecord] = []
    if not path.exists():
        return records
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(RolloutRecord.model_validate(json.loads(line)))
    return records


@dataclass(frozen=True)
class V2SeedResults:
    seed_index: int
    seed_dir: Path

    def rollouts(self) -> list[RolloutRecord]:
        return _parse_rollouts(self.seed_dir / "rollouts.jsonl")

    def successful_rollouts(self) -> list[RolloutRecord]:
        return [
            r for r in self.rollouts() if isinstance(r.outcome, SuccessFeedback)
        ]

    def best_by_step(self) -> list[BestByStep]:
        """Running-best reward per step (monotone non-decreasing).

        Steps with no successful rollout before them carry the previous
        best; steps whose successful rollouts don't beat the running best
        carry the prior leader's candidate id.
        """
        out: list[BestByStep] = []
        current_best_reward = float("-inf")
        current_best_id: str | None = None
        records_by_step: dict[int, list[RolloutRecord]] = {}
        for r in self.successful_rollouts():
            records_by_step.setdefault(r.step, []).append(r)
        if not records_by_step:
            return out
        for step in sorted(records_by_step.keys()):
            for r in records_by_step[step]:
                if r.reward > current_best_reward:
                    current_best_reward = r.reward
                    current_best_id = r.candidate_id
            if current_best_id is not None:
                out.append(
                    BestByStep(
                        step=step,
                        best_reward=current_best_reward,
                        best_candidate_id=current_best_id,
                    )
                )
        return out

    def summary(self) -> SeedSummary:
        rollouts = self.rollouts()
        counts = {
            "success": 0,
            "parse_failure": 0,
            "compile_failed": 0,
            "runtime_error": 0,
            "incorrect": 0,
            "infrastructure_failure": 0,
        }
        best_reward: float | None = None
        for r in rollouts:
            counts[r.outcome.kind] = counts.get(r.outcome.kind, 0) + 1
            if isinstance(r.outcome, SuccessFeedback):
                if best_reward is None or r.reward > best_reward:
                    best_reward = r.reward
        return SeedSummary(
            seed_index=self.seed_index,
            num_rollouts=len(rollouts),
            num_successes=counts["success"],
            num_parse_failures=counts["parse_failure"],
            num_compile_failures=counts["compile_failed"],
            num_runtime_errors=counts["runtime_error"],
            num_incorrect=counts["incorrect"],
            num_infra_failures=counts["infrastructure_failure"],
            best_reward=best_reward,
        )


class V2ExperimentResults:
    _output_dir: Path

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir

    @property
    def output_dir(self) -> Path:
        return self._output_dir

    def config_path(self) -> Path:
        return self._output_dir / "experiment.json"

    def per_seed(self) -> list[V2SeedResults]:
        tinker_log = self._output_dir / "tinker_log"
        if not tinker_log.exists():
            return []
        out: list[V2SeedResults] = []
        for seed_dir in sorted(tinker_log.glob("seed_*")):
            seed_index = int(seed_dir.name.removeprefix("seed_"))
            out.append(V2SeedResults(seed_index=seed_index, seed_dir=seed_dir))
        return out
