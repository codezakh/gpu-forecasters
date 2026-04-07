"""Invocation sink: lightweight cost-tracking interface for providers.

Each provider (evaluator or mutator) accepts an optional ``InvocationSink``.
When present, the provider writes a structured record after each invocation.
When ``None``, no tracking occurs and there is zero overhead.

Tying records back to search nodes
------------------------------------
Records are tied to ``Node`` objects via ``code_sha256``, a SHA-256 digest of
the program code string.  Because ``Node.program_code`` is immutable and PUCT
deduplicates on program code content, the same hash uniquely identifies a node
within a run.  The join is performed after the fact during analysis — no
coupling between the sink and the search algorithm is needed.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel
from ulid import ULID


@runtime_checkable
class InvocationSink(Protocol):
    """Receives and persists provider invocation records."""

    def record(self, payload: BaseModel) -> None: ...


class FilesystemInvocationSink:
    """Writes each invocation record as an individual JSON file.

    Files are named ``<ULID>.json``, which guarantees:

    - No filename collisions across concurrent writers.
    - No file-level locking required (one file per record).
    - Natural chronological ordering (ULIDs are time-sortable).
    """

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._directory.mkdir(parents=True, exist_ok=True)

    def record(self, payload: BaseModel) -> None:
        record_id = str(ULID())
        path = self._directory / f"{record_id}.json"
        path.write_text(payload.model_dump_json(indent=2))


def code_sha256(program_code: str) -> str:
    """Return the SHA-256 hex digest of *program_code* (UTF-8 encoded)."""
    return hashlib.sha256(program_code.encode("utf-8")).hexdigest()
