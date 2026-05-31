"""Invocation sink: lightweight cost-tracking interface for providers.

Each provider (evaluator or mutator) accepts an optional ``InvocationSink``.
When present, the provider writes a structured record after each invocation.
When ``None``, no tracking occurs and there is zero overhead.

**Fault tolerance contract**
-----------------------------
``InvocationSink`` implementations MUST NOT raise exceptions under any
circumstances.  Sinks are optional cost-tracking infrastructure; a transient
write failure (disk full, permissions error, etc.) must never propagate up and
abort the provider call that triggered it.  Implementations should log a
warning and swallow the error.

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

from loguru import logger
from pydantic import BaseModel
from ulid import ULID


@runtime_checkable
class InvocationSink(Protocol):
    """Receives and persists provider invocation records.

    Implementations must never raise — see module docstring.
    """

    def record(self, payload: BaseModel) -> None: ...


class FilesystemInvocationSink:
    """Writes each invocation record as an individual JSON file.

    Files are named ``<ULID>.json``, which guarantees:

    - No filename collisions across concurrent writers.
    - No file-level locking required (one file per record).
    - Natural chronological ordering (ULIDs are time-sortable).

    Write failures are caught and logged as warnings; they never propagate.
    """

    def __init__(self, directory: Path) -> None:
        self._directory = directory
        self._directory.mkdir(parents=True, exist_ok=True)

    def record(self, payload: BaseModel) -> None:
        try:
            record_id = str(ULID())
            path = self._directory / f"{record_id}.json"
            path.write_text(payload.model_dump_json(indent=2))
        except Exception:
            logger.opt(exception=True).warning(
                "FilesystemInvocationSink failed to write record; "
                "cost tracking may be incomplete."
            )


def code_sha256(program_code: str) -> str:
    """Return the SHA-256 hex digest of *program_code* (UTF-8 encoded)."""
    return hashlib.sha256(program_code.encode("utf-8")).hexdigest()
