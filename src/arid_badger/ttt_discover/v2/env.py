"""The v2 rollout environment.

Wires together: task + feedback prompt renderers, a Tinker chat-template
``Renderer`` for the model's chat format, a ``KernelEvaluator``, a
``RewardScalarizer``, a ``CodeExtractor``, a ``CandidateArchive``, and a
``RolloutSink``. Emits exactly one ``RolloutRecord`` per episode.

Unlike v1, the environment is *not* responsible for managing search
state — the ``CandidateArchive`` is. The env receives its parent
candidate via the constructor (the rl_integration layer samples it once
per group and fans out to ``group_size`` envs), evaluates one child, and
inserts the child into the archive.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timezone

import tinker

from arid_badger.ttt_discover.v1.rl.types import Action, StepResult
from arid_badger.ttt_discover.v1.tinker_utils import renderers as v1_renderers
from arid_badger.ttt_discover.v2.archive.puct import build_candidate
from arid_badger.ttt_discover.v2.domain.candidate import Candidate
from arid_badger.ttt_discover.v2.domain.context import (
    FeedbackPromptContext,
    TaskPromptContext,
)
from arid_badger.ttt_discover.v2.domain.outcome import (
    ParseFailureFeedback,
    TriMulRLOutcome,
)
from arid_badger.ttt_discover.v2.domain.problem import TriMulProblem
from arid_badger.ttt_discover.v2.domain.records import RolloutRecord
from arid_badger.ttt_discover.v2.interfaces.archive import CandidateArchive
from arid_badger.ttt_discover.v2.interfaces.evaluator import KernelEvaluator
from arid_badger.ttt_discover.v2.interfaces.extractor import CodeExtractor
from arid_badger.ttt_discover.v2.interfaces.renderer import (
    FeedbackPromptRenderer,
    TaskPromptRenderer,
)
from arid_badger.ttt_discover.v2.interfaces.scalarizer import RewardScalarizer
from arid_badger.ttt_discover.v2.interfaces.sink import RolloutSink


class TriMulRLEnvironment:
    """Implements v1's ``ProblemEnv`` interface — see
    ``arid_badger.ttt_discover.v1.rl.types.Env``. Does not subclass it;
    v1's ``ProblemEnv`` has abstract methods (``check_answer`` etc.)
    whose contracts v2 doesn't honour, so duck-typing via ``Env``
    (which only requires ``initial_observation`` + ``step``) is cleaner.
    """

    _problem: TriMulProblem
    _task_renderer: TaskPromptRenderer
    _feedback_renderer: FeedbackPromptRenderer
    _tinker_renderer: v1_renderers.Renderer
    _evaluator: KernelEvaluator
    _scalarizer: RewardScalarizer
    _extractor: CodeExtractor
    _archive: CandidateArchive
    _sink: RolloutSink
    _parent: Candidate | None
    _timestep: int
    _group_index: int
    _rollout_index: int

    # Populated by initial_observation() so step() can record them on the
    # RolloutRecord without re-rendering.
    _task_prompt: str
    _feedback_prompt: str
    _sampling_start_s: float

    def __init__(
        self,
        *,
        problem: TriMulProblem,
        task_prompt_renderer: TaskPromptRenderer,
        feedback_prompt_renderer: FeedbackPromptRenderer,
        tinker_renderer: v1_renderers.Renderer,
        evaluator: KernelEvaluator,
        scalarizer: RewardScalarizer,
        extractor: CodeExtractor,
        archive: CandidateArchive,
        sink: RolloutSink,
        parent: Candidate | None,
        timestep: int,
        group_index: int,
        rollout_index: int,
    ) -> None:
        self._problem = problem
        self._task_renderer = task_prompt_renderer
        self._feedback_renderer = feedback_prompt_renderer
        self._tinker_renderer = tinker_renderer
        self._evaluator = evaluator
        self._scalarizer = scalarizer
        self._extractor = extractor
        self._archive = archive
        self._sink = sink
        self._parent = parent
        self._timestep = timestep
        self._group_index = group_index
        self._rollout_index = rollout_index

        self._task_prompt = ""
        self._feedback_prompt = ""
        self._sampling_start_s = 0.0

    @property
    def stop_condition(self) -> list[str] | list[int]:
        return self._tinker_renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[tinker.ModelInput, list[str] | list[int]]:
        task_ctx = TaskPromptContext(
            problem=self._problem,
            archive=self._archive,
            parent=self._parent,
            timestep=self._timestep,
        )
        feedback_ctx = FeedbackPromptContext(
            problem=self._problem,
            parent=self._parent,
        )
        self._task_prompt = self._task_renderer.render(task_ctx)
        self._feedback_prompt = self._feedback_renderer.render(feedback_ctx)

        user_content = self._task_prompt
        if self._feedback_prompt:
            user_content = f"{self._task_prompt}\n\n{self._feedback_prompt}"

        convo: list[v1_renderers.Message] = [
            {"role": "user", "content": user_content}
        ]
        self._sampling_start_s = time.time()
        model_input = self._tinker_renderer.build_generation_prompt(convo)
        return model_input, self.stop_condition

    async def step(self, action: Action, *args: object, **kwargs: object) -> StepResult:
        sampling_time_s = time.time() - self._sampling_start_s
        prompt_tokens = len(self._task_prompt) + len(self._feedback_prompt)
        response_tokens = len(action)

        message, parse_success = self._tinker_renderer.parse_response(action)
        raw_response = v1_renderers.ensure_text(message["content"])

        outcome: TriMulRLOutcome
        parsed_code: str | None
        eval_start_s = time.time()
        if not parse_success:
            parsed_code = None
            outcome = ParseFailureFeedback(reason="renderer.parse_response returned parse_success=False")
        else:
            parsed_code = self._extractor.extract(raw_response)
            if parsed_code is None or not parsed_code.strip():
                outcome = ParseFailureFeedback(
                    reason="no extractable python code block in response"
                )
                parsed_code = None
            else:
                outcome = await self._evaluator.evaluate(parsed_code)
        eval_time_s = time.time() - eval_start_s

        reward = self._scalarizer.scalarize(outcome)

        # Archive update + candidate creation.
        if parsed_code is not None and outcome.kind == "success":
            candidate = build_candidate(
                code=parsed_code,
                timestep=self._timestep,
                parent_id=self._parent.id if self._parent is not None else None,
                outcome=outcome,
                reward=reward,
            )
            self._archive.insert(candidate, self._parent)
        else:
            # Failed rollouts: still need a candidate_id for the record
            # (so downstream joins can key off it) but we don't insert
            # them into the archive (they aren't useful search state).
            candidate = build_candidate(
                code=parsed_code or "",
                timestep=self._timestep,
                parent_id=self._parent.id if self._parent is not None else None,
                outcome=outcome,
                reward=reward,
            )
            self._archive.record_failed_attempt(self._parent)

        record = RolloutRecord(
            step=self._timestep,
            group_index=self._group_index,
            rollout_index=self._rollout_index,
            timestamp_utc=datetime.now(timezone.utc).isoformat(),
            parent_id=self._parent.id if self._parent is not None else None,
            candidate_id=candidate.id,
            task_prompt=self._task_prompt,
            feedback_prompt=self._feedback_prompt,
            raw_response=raw_response,
            parsed_code=parsed_code,
            outcome=outcome,
            reward=reward,
            prompt_tokens=prompt_tokens,
            response_tokens=response_tokens,
            sampling_time_s=sampling_time_s,
            eval_time_s=eval_time_s,
        )
        # The sink does file I/O; offload to a thread so a slow write
        # doesn't block the rollout loop's event loop.
        await asyncio.to_thread(self._sink.record, record)

        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self.stop_condition,
            metrics={
                "reward": reward,
                "parse_success": float(parse_success),
                "eval_time_s": eval_time_s,
                "sampling_time_s": sampling_time_s,
            },
        )
