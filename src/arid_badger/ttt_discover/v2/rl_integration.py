"""RL dataset adapter that surfaces v2 components to v1's training loop.

v1's ``do_sync_training`` expects an ``RLDataset`` that yields
``EnvGroupBuilder`` batches; each builder's ``make_envs`` returns a group
of ``Env`` instances with a shared starting state. In v2 the "shared
starting state" is a parent ``Candidate`` selected by the archive.

One ``V2RLDataset.get_batch(i)`` call therefore:

1. Samples ``groups_per_batch`` parent candidates from the archive.
2. For each parent, builds a ``V2ProblemGroupBuilder`` that will in turn
   build ``group_size`` ``TriMulRLEnvironment`` instances pinned to that
   parent. The env's ``group_index`` / ``rollout_index`` identify the
   rollout in the event log.

``flush(step)`` snapshots the archive. Checkpoint recovery is not yet
implemented in v2 — a re-run starts from an empty archive. (The event
log is still recoverable from ``rollouts.jsonl`` alone.)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import chz

from arid_badger.ttt_discover.v1.rl.types import (
    Env,
    EnvGroupBuilder,
    Metrics,
    RLDataset,
    RLDatasetBuilder,
    Trajectory,
)
from arid_badger.ttt_discover.v1.tinker_utils import renderers as v1_renderers
from arid_badger.ttt_discover.v2.domain.candidate import Candidate
from arid_badger.ttt_discover.v2.domain.problem import TriMulProblem
from arid_badger.ttt_discover.v2.env import TriMulRLEnvironment
from arid_badger.ttt_discover.v2.interfaces.archive import CandidateArchive
from arid_badger.ttt_discover.v2.interfaces.evaluator import KernelEvaluator
from arid_badger.ttt_discover.v2.interfaces.extractor import CodeExtractor
from arid_badger.ttt_discover.v2.interfaces.renderer import (
    FeedbackPromptRenderer,
    TaskPromptRenderer,
)
from arid_badger.ttt_discover.v2.interfaces.scalarizer import RewardScalarizer
from arid_badger.ttt_discover.v2.interfaces.sink import RolloutSink


@dataclass(frozen=True)
class V2Components:
    problem: TriMulProblem
    task_prompt_renderer: TaskPromptRenderer
    feedback_prompt_renderer: FeedbackPromptRenderer
    evaluator: KernelEvaluator
    scalarizer: RewardScalarizer
    extractor: CodeExtractor
    archive: CandidateArchive
    sink: RolloutSink


@dataclass(frozen=True)
class V2ProblemGroupBuilder(EnvGroupBuilder):
    components: V2Components
    tinker_renderer: v1_renderers.Renderer
    parent: Candidate | None
    timestep: int
    group_index: int
    group_size: int
    logging_name: str = "trimul_v2"

    async def make_envs(self) -> Sequence[Env]:
        envs: list[Env] = []
        for rollout_index in range(self.group_size):
            env = TriMulRLEnvironment(
                problem=self.components.problem,
                task_prompt_renderer=self.components.task_prompt_renderer,
                feedback_prompt_renderer=self.components.feedback_prompt_renderer,
                tinker_renderer=self.tinker_renderer,
                evaluator=self.components.evaluator,
                scalarizer=self.components.scalarizer,
                extractor=self.components.extractor,
                archive=self.components.archive,
                sink=self.components.sink,
                parent=self.parent,
                timestep=self.timestep,
                group_index=self.group_index,
                rollout_index=rollout_index,
            )
            envs.append(env)  # pyright: ignore[reportArgumentType]
        return envs

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> list[str]:
        return [self.logging_name]


class V2RLDataset(RLDataset):
    _components: V2Components
    _tinker_renderer: v1_renderers.Renderer
    _groups_per_batch: int
    _group_size: int

    def __init__(
        self,
        *,
        components: V2Components,
        tinker_renderer: v1_renderers.Renderer,
        groups_per_batch: int,
        group_size: int,
    ) -> None:
        self._components = components
        self._tinker_renderer = tinker_renderer
        self._groups_per_batch = groups_per_batch
        self._group_size = group_size

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        parents = self._components.archive.sample(self._groups_per_batch)
        # archive.sample may return fewer than requested if the archive is
        # still small — pad with ``None`` (cold-start) so every batch has
        # exactly ``groups_per_batch`` groups.
        while len(parents) < self._groups_per_batch:
            parents.append(None)  # pyright: ignore[reportArgumentType]

        builders: list[EnvGroupBuilder] = []
        for group_index, parent in enumerate(parents):
            builders.append(
                V2ProblemGroupBuilder(
                    components=self._components,
                    tinker_renderer=self._tinker_renderer,
                    parent=parent,
                    timestep=index,
                    group_index=group_index,
                    group_size=self._group_size,
                )
            )
        return builders

    def flush(self, step: int | None = None) -> None:
        self._components.archive.snapshot(step if step is not None else 0)

    def __len__(self) -> int:
        return 1


@chz.chz
class V2RLDatasetBuilder(RLDatasetBuilder):
    components: V2Components
    model_name_for_tokenizer: str
    renderer_name: str
    groups_per_batch: int
    group_size: int

    async def __call__(self) -> V2RLDataset:  # pyright: ignore[reportIncompatibleMethodOverride]
        from arid_badger.ttt_discover.v1.tinker_utils.misc_utils import get_tokenizer

        tokenizer = get_tokenizer(self.model_name_for_tokenizer)
        tinker_renderer = v1_renderers.get_renderer(
            self.renderer_name, tokenizer=tokenizer
        )
        return V2RLDataset(
            components=self.components,
            tinker_renderer=tinker_renderer,
            groups_per_batch=self.groups_per_batch,
            group_size=self.group_size,
        )
