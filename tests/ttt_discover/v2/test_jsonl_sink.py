import json
from pathlib import Path

from gpu_forecasters.trimul.core import IncorrectFeedback
from gpu_forecasters.ttt_discover.v2.domain.candidate import CandidateId
from gpu_forecasters.ttt_discover.v2.domain.records import RolloutRecord
from gpu_forecasters.ttt_discover.v2.sinks.jsonl import (
    JsonlRolloutSink,
    ListRolloutSink,
)


def _make_record(i: int) -> RolloutRecord:
    return RolloutRecord(
        step=i,
        group_index=0,
        rollout_index=0,
        timestamp_utc="2026-04-24T00:00:00+00:00",
        parent_id=None,
        candidate_id=CandidateId(f"c{i}"),
        task_prompt="t",
        feedback_prompt="",
        raw_response="r",
        parsed_code=None,
        outcome=IncorrectFeedback(error_message=f"err {i}"),
        reward=0.0,
        prompt_tokens=0,
        response_tokens=0,
        sampling_time_s=0.0,
        eval_time_s=0.0,
    )


def test_jsonl_sink_writes_one_line_per_record(tmp_path: Path) -> None:
    path = tmp_path / "rollouts.jsonl"
    sink = JsonlRolloutSink(path=path)
    originals = [_make_record(i) for i in range(3)]
    for r in originals:
        sink.record(r)
    sink.close()

    lines = path.read_text().splitlines()
    assert len(lines) == 3
    for line, original in zip(lines, originals, strict=True):
        parsed = RolloutRecord.model_validate(json.loads(line))
        assert parsed == original


def test_jsonl_sink_is_append_only(tmp_path: Path) -> None:
    path = tmp_path / "rollouts.jsonl"

    JsonlRolloutSink(path=path).record(_make_record(0))
    JsonlRolloutSink(path=path).record(_make_record(1))

    lines = path.read_text().splitlines()
    assert len(lines) == 2


def test_list_sink_collects() -> None:
    sink = ListRolloutSink()
    for i in range(4):
        sink.record(_make_record(i))
    assert len(sink.records) == 4
    assert sink.records[-1].step == 3
