"""The durable per-rollout event record.

Written to ``rollouts.jsonl`` as the primary artifact of a v2 run; the
results loader reads nothing else. Every field needed for downstream
analysis of a single rollout is present on the record — no cross-
referencing a separate archive snapshot or metrics file.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from arid_badger.ttt_discover.v2.domain.candidate import CandidateId
from arid_badger.ttt_discover.v2.domain.outcome import TriMulRLOutcome


class RolloutRecord(BaseModel):
    model_config = ConfigDict(frozen=True)

    step: int
    group_index: int
    rollout_index: int
    timestamp_utc: str

    parent_id: CandidateId | None
    candidate_id: CandidateId

    task_prompt: str
    feedback_prompt: str

    # Full decoded assistant response — includes CoT / analysis-channel
    # tokens for reasoning models. This is the field v1 silently dropped
    # via ``remove_non_numerical_field`` before ``metrics.jsonl``; keep it
    # as a first-class column here.
    raw_response: str
    parsed_code: str | None
    outcome: TriMulRLOutcome
    reward: float

    prompt_tokens: int
    response_tokens: int
    sampling_time_s: float
    eval_time_s: float
