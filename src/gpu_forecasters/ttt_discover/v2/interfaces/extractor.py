from __future__ import annotations

from typing import Protocol


class CodeExtractor(Protocol):
    def extract(self, raw_response: str) -> str | None: ...
