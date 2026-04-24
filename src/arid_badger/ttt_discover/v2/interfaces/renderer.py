from __future__ import annotations

from typing import Protocol

from arid_badger.ttt_discover.v2.domain.context import (
    FeedbackPromptContext,
    TaskPromptContext,
)


class TaskPromptRenderer(Protocol):
    def render(self, ctx: TaskPromptContext) -> str: ...


class FeedbackPromptRenderer(Protocol):
    def render(self, ctx: FeedbackPromptContext) -> str: ...
