"""End-to-end ``TriMulRLEnvironment.step()`` with fake collaborators.

Exercises the real wiring (context building, prompt rendering, extractor,
scalarizer, archive insert, sink record) through a single rollout and
asserts the observable side-effects: sink got a ``RolloutRecord``, the
``Candidate`` made it into the archive, and ``StepResult.reward`` matches
the scalarizer output.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import cast

import pytest
import tinker

from arid_badger.trimul.core import (
    CaseSpeedup,
    IncorrectFeedback,
    SuccessFeedback,
)
from arid_badger.ttt_discover.v2.archive.puct import PUCTCandidateArchive
from arid_badger.ttt_discover.v2.domain.outcome import TriMulRLOutcome
from arid_badger.ttt_discover.v2.domain.problem import TriMulProblem
from arid_badger.ttt_discover.v2.env import TriMulRLEnvironment
from arid_badger.ttt_discover.v2.extractors.python_block import (
    LastPythonBlockExtractor,
)
from arid_badger.ttt_discover.v2.renderers.feedback_trimul import (
    TriMulFeedbackPromptRenderer,
)
from arid_badger.ttt_discover.v2.renderers.task_static import (
    StaticTaskPromptRenderer,
)
from arid_badger.ttt_discover.v2.scalarizers.by_target_us import ScaleByTargetUs
from arid_badger.ttt_discover.v2.sinks.jsonl import ListRolloutSink


class FakeEvaluator:
    def __init__(self, outcome: TriMulRLOutcome) -> None:
        self._outcome = outcome
        self.calls: list[str] = []

    async def evaluate(self, code: str) -> TriMulRLOutcome:
        self.calls.append(code)
        return self._outcome


class FakeTinkerRenderer:
    """Minimal renderer — round-trips ``build_generation_prompt`` +
    ``parse_response`` without invoking a real tokenizer."""

    def __init__(self, response_text: str, parse_success: bool = True) -> None:
        self._response_text = response_text
        self._parse_success = parse_success
        self.last_messages: list[dict[str, object]] | None = None

    def get_stop_sequences(self) -> list[str]:
        return ["<|endoftext|>"]

    def build_generation_prompt(
        self, messages, role: str = "assistant", prefill: str | None = None
    ) -> tinker.ModelInput:
        self.last_messages = list(messages)
        return tinker.ModelInput.empty()

    def parse_response(self, action):
        # The action is list[int]; we ignore it and return our canned response.
        return ({"role": "assistant", "content": self._response_text}, self._parse_success)


def _problem() -> TriMulProblem:
    return TriMulProblem(
        base_prompt_text="BASE PROMPT",
        test_cases=(),
        gpu_name="A100-80GB",
        triton_version="3.3.1",
        target_runtime_us=2500.0,
    )


def _success_outcome() -> SuccessFeedback:
    return SuccessFeedback(
        aggregated_speedup=1.0,
        aggregation_method="geomean",
        per_case_speedups=[
            CaseSpeedup(
                seqlen=256,
                bs=2,
                dim=128,
                hiddendim=128,
                nomask=True,
                distribution="normal",
                speedup=1.0,
                runtime_ns=2_500_000.0,
                ref_runtime_ns=2_500_000.0,
            )
        ],
    )


def _make_env(
    *,
    tmp_path: Path,
    response_text: str,
    outcome: TriMulRLOutcome,
    parse_success: bool = True,
) -> tuple[TriMulRLEnvironment, ListRolloutSink, PUCTCandidateArchive, FakeEvaluator]:
    problem = _problem()
    archive = PUCTCandidateArchive(directory=tmp_path)
    sink = ListRolloutSink()
    evaluator = FakeEvaluator(outcome=outcome)
    tinker_renderer = FakeTinkerRenderer(
        response_text=response_text, parse_success=parse_success
    )
    env = TriMulRLEnvironment(
        problem=problem,
        task_prompt_renderer=StaticTaskPromptRenderer(),
        feedback_prompt_renderer=TriMulFeedbackPromptRenderer(),
        tinker_renderer=cast("object", tinker_renderer),  # pyright: ignore[reportArgumentType]
        evaluator=evaluator,
        scalarizer=ScaleByTargetUs(target_us=2500.0),
        extractor=LastPythonBlockExtractor(),
        archive=archive,
        sink=sink,
        parent=None,
        timestep=7,
        group_index=2,
        rollout_index=3,
    )
    return env, sink, archive, evaluator


def _run(env: TriMulRLEnvironment) -> None:
    async def _go() -> None:
        _ = await env.initial_observation()
        _ = await env.step([1, 2, 3])

    asyncio.run(_go())


def test_successful_rollout_emits_record_and_inserts_candidate(tmp_path: Path) -> None:
    response = "here's the kernel\n```python\ndef custom_kernel(data): return data[0]\n```"
    env, sink, archive, evaluator = _make_env(
        tmp_path=tmp_path, response_text=response, outcome=_success_outcome()
    )
    _run(env)

    assert len(sink.records) == 1
    record = sink.records[0]
    assert record.step == 7
    assert record.group_index == 2
    assert record.rollout_index == 3
    assert record.outcome.kind == "success"
    assert record.reward == pytest.approx(1.0)
    assert record.parsed_code == "def custom_kernel(data): return data[0]"
    assert record.raw_response == response
    assert len(evaluator.calls) == 1

    # Archive now has root + the inserted child.
    picked = archive.sample(n=2)
    codes = {c.code for c in picked}
    assert "def custom_kernel(data): return data[0]" in codes or picked[0].code == (
        "def custom_kernel(data): return data[0]"
    )


def test_parse_failure_when_no_codeblock(tmp_path: Path) -> None:
    env, sink, _archive, evaluator = _make_env(
        tmp_path=tmp_path,
        response_text="nope, no block",
        outcome=_success_outcome(),  # never reached
    )
    _run(env)

    assert len(sink.records) == 1
    record = sink.records[0]
    assert record.outcome.kind == "parse_failure"
    assert record.reward == 0.0
    assert record.parsed_code is None
    # Evaluator should not have been called.
    assert evaluator.calls == []


def test_failed_rollout_skips_archive_insert_but_logs_record(tmp_path: Path) -> None:
    env, sink, _archive, _evaluator = _make_env(
        tmp_path=tmp_path,
        response_text="```python\nthis will be called incorrect\n```",
        outcome=IncorrectFeedback(error_message="max abs err 1"),
    )
    _run(env)

    assert sink.records[0].outcome.kind == "incorrect"
    assert sink.records[0].reward == 0.0
