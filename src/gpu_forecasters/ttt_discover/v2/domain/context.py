"""Prompt-rendering contexts.

``TaskPromptContext`` and ``FeedbackPromptContext`` are kept as separate
dataclasses because they change for different reasons: the task prompt
may start depending on live archive state (e.g. "focus on the slowest
case from the archive's current best"), while the feedback prompt is
fully determined by the parent candidate's outcome.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from gpu_forecasters.ttt_discover.v2.domain.candidate import Candidate
from gpu_forecasters.ttt_discover.v2.domain.problem import TriMulProblem

if TYPE_CHECKING:
    from gpu_forecasters.ttt_discover.v2.interfaces.archive import CandidateArchive


@dataclass(frozen=True)
class TaskPromptContext:
    problem: TriMulProblem
    archive: "CandidateArchive"
    parent: Candidate | None
    timestep: int


@dataclass(frozen=True)
class FeedbackPromptContext:
    problem: TriMulProblem
    parent: Candidate | None
